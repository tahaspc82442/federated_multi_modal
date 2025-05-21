"""dualprompt original with renaming and visualization
"""

import copy
import torch
import torch.nn.functional as F
from collections import defaultdict, Counter, OrderedDict # Ensure OrderedDict is imported
from tqdm import tqdm, trange # For progress bars during evaluation
import wandb
import copy
import os
import hashlib
import os.path as osp
import numpy as np
from PIL import Image


from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.data import DataManager
from dassl.data.datasets import Datum
from dassl.utils import (
    mkdir_if_missing,
    load_checkpoint,
    save_checkpoint
)
from dassl.optim import build_lr_scheduler, build_optimizer # Added build_optimizer potentially needed
from torch.utils.data import DataLoader # Needed for manual loader creation

# Local imports (adjust paths if necessary)
from trainers.dualprompt import DualPrompt, load_clip_to_cpu
from .client_datamanager import ClientDataManager
from .debug import debug_collate # Assuming this is a custom collate function


# Helper function to apply class merges to Datum objects within a dataset split
def apply_merge_to_datums(data_list, source_to_target_map):
    """
    Updates the classname attribute of Datum objects based on the merge map.
    Does NOT change the label attribute here.
    """
    if not data_list: # Handle empty lists (e.g., empty val set)
        return
    updated_count = 0
    for i in range(len(data_list)):
        item = data_list[i]
        # Ensure classname exists and is a string
        if not hasattr(item, 'classname') or not isinstance(item.classname, str):
             print(f"Warning: Datum at index {i} has invalid classname: {getattr(item, 'classname', 'Missing')}. Skipping merge.")
             continue

        original_cname_lower = item.classname.lower()
        if original_cname_lower in source_to_target_map:
            target_cname = source_to_target_map[original_cname_lower]
            # Only update if the name actually changes
            if item.classname != target_cname:
                # Create new Datum, preserving other attributes
                new_datum_args = {
                    "impath": item.impath,
                    "label": item.label, # Keep original label for now
                    "classname": target_cname # Update classname to target
                }
                # Preserve optional attributes if they exist
                for attr in ['domain', 'caption']:
                    if hasattr(item, attr):
                        new_datum_args[attr] = getattr(item, attr)

                data_list[i] = Datum(**new_datum_args)
                updated_count += 1
    #if updated_count > 0:
    #    print(f"    Updated {updated_count} Datum classnames based on merge rules.") # Less verbose

# Helper function to update lab2cname dictionary based on merges
def update_lab2cname_for_merge(lab2cname, source_to_target_map):
    """Updates the lab2cname mapping to reflect merged classes."""
    updated_count = 0
    labels_to_update = list(lab2cname.keys()) # Iterate over a copy of keys
    for label in labels_to_update:
        original_cname = lab2cname.get(label) # Use .get for safety
        if original_cname and isinstance(original_cname, str):
             original_cname_lower = original_cname.lower()
             if original_cname_lower in source_to_target_map:
                 target_cname = source_to_target_map[original_cname_lower]
                 # Only update if the name actually changes
                 if lab2cname[label] != target_cname:
                     lab2cname[label] = target_cname
                     updated_count += 1
    #if updated_count > 0:
    #    print(f"    Updated {updated_count} lab2cname entries based on merge rules.") # Less verbose


@TRAINER_REGISTRY.register()
class DualPromptFL(TrainerX):
    def __init__(self, cfg):
        # Must define self.lab2cname before super().__init__(cfg),
        # because Dassl might build an evaluator in TrainerX.__init__
        self.lab2cname = {} # Initialize empty, will be populated in build_data_loader
        self._clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.DUALPROMPT.PREC == "fp16":
            self._clip_model = self._clip_model.half()
        self._clip_model = self._clip_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.cfg = cfg
        self.num_clients = cfg.FED.NUM_CLIENTS
        self.num_rounds = cfg.FED.NUM_ROUNDS
        self.local_epochs = cfg.FED.LOCAL_EPOCHS
        self.clients = []
        self.global_weights = None
        self.prox_mu = 0.0 #cfg.FED.PROX_MU

        self.nan_stats = {
            "total_updates": 0,
            "failed_clients": [],
            "skipped_rounds": 0
        }
        # Make sure super init happens after essential attributes are set
        super().__init__(cfg)

    ###################################################
    # A) Build the unified data loader
    ###################################################
    def build_data_loader(self):
        print("build data loader called")
        # ... (Dataset Loading as before) ...
        print("Loading datasets...")
        pat_cfg = self.cfg.clone(); pat_cfg.defrost(); pat_cfg.DATASET.NAME = "PatternNet"; dm_pn = DataManager(pat_cfg); dataset_pn = dm_pn.dataset; self.dataset_pn = dataset_pn
        uc_cfg = self.cfg.clone(); uc_cfg.defrost(); uc_cfg.DATASET.NAME = "Ucmerced"; dm_uc = DataManager(uc_cfg); dataset_uc = dm_uc.dataset; self.dataset_uc = dataset_uc
        euro_cfg = self.cfg.clone(); euro_cfg.defrost(); euro_cfg.DATASET.NAME = "EuroSAT"; dm_euro = DataManager(euro_cfg); dataset_euro = dm_euro.dataset; self.dataset_euro = dataset_euro
        mlrs_cfg=self.cfg.clone(); mlrs_cfg.defrost(); mlrs_cfg.DATASET.NAME="Mlrs"; dm_mlrs=DataManager(mlrs_cfg); dataset_mlrs=dm_mlrs.dataset; self.dataset_mlrs=dataset_mlrs
        milaid_cfg=self.cfg.clone(); milaid_cfg.defrost(); milaid_cfg.DATASET.NAME="Milaid"; dm_milaid=DataManager(milaid_cfg); dataset_milaid=dm_milaid.dataset; self.dataset_milaid=dataset_milaid
        print("Datasets loaded.")

        # Get initial lab2cname mappings
        pn_lab2cname = dm_pn.lab2cname
        uc_lab2cname = dm_uc.lab2cname
        euro_lab2cname = dm_euro.lab2cname
        mlrs_lab2cname=dm_mlrs.lab2cname
        milaid_lab2cname=dm_milaid.lab2cname

        # -- 2) Initial Class Name Normalization --
        print("Performing initial class name normalization...")
        uc_rename_map = {
            "tenniscourt": "tennis_court", "golfcourse": "golf_course",
            "parkinglot": "parking_lot", "storagetanks": "storage_tank",
            "mobilehomepark": "mobile_home_park", "baseballdiamond": "baseball_field", # Merged later
            "denseresidential": "dense_residential", "sparseresidential": "sparse_residential"
        }
        for k, old_cname in uc_lab2cname.items():
            if old_cname in uc_rename_map: uc_lab2cname[k] = uc_rename_map[old_cname]

        milaid_rename_map = {
            "commercial area": "commercial_area", "ice land": "ice_land",
            "bare land": "bare_land", "detached house": "detached_house",
            "dry field": "dry_field", "golf course": "golf_course",
            "ground track field": "ground_track_field", "mobile home park": "mobile_home_park",
            "oil field": "oil_field", "paddy field": "paddy_field",
            "parking lot": "parking_lot", "rock land": "rock_land",
            "solar power plant": "solar_power_plant", "sparse shrub land": "sparse_shrub_land",
            "storage tank": "storage_tank", "swimming pool": "swimming_pool",
            "terraced field": "terraced_field", "train station": "train_station", # Merged later
            "wastewater plant": "wastewater_plant", "wind turbine": "wind_turbine",
            "baseball field": "baseball_field", "basketball court": "basketball_court",
            "tennis court": "tennis_court",
        }
        for label, old_cname in milaid_lab2cname.items():
            if old_cname in milaid_rename_map: milaid_lab2cname[label] = milaid_rename_map[old_cname]
        print("Initial normalization complete.")


        # -- 3) Define and Apply Class Merges --
        print("Defining and applying class merges...")
        merge_rules = [
            (['bare_land', 'bareland'], 'bare_land'),
            (['dense_residential', 'dense_residential_area'], 'dense_residential_area'),
            (['sparse_residential', 'sparse_residential_area'], 'sparse_residential_area'),
            (['harbor', 'harbor_port'], 'harbor_port'),
            (['runway', 'runway_marking'], 'runway'),
            (['railway_station', 'train_station'], 'railway_station'),
            (['substation', 'transformer_station'], 'substation'),
            (['wastewater_plant', 'wastewater_treatment_plant'], 'wastewater_treatment_plant'),
            (['terrace', 'terraced_field'], 'terraced_field'),
            (['parking_lot', 'parking_space'], 'parking_area'),
            (['greenhouse', 'vegetable_greenhouse'], 'greenhouse'),
            (['oil_field', 'oil_gas_field', 'oil_well'], 'oil_gas_field'),
            (['mine', 'quarry'], 'mine_or_quarry'),
            (['baseball_diamond', 'baseball_field'], 'baseball_field'),
            (['solar_panel', 'solar_power_plant'], 'solar_power_plant'),
            (['bridge', 'overpass', 'viaduct'], 'elevated_crossing'),
            (['industrial_buildings', 'industrial_area'], 'industrial_area'),
            (['freeway', 'parkway', 'road'], 'road'),
            (['works', 'buildings'], 'buildings')
        ]
        source_to_target_map = {s.lower(): t.lower() for sources, t in merge_rules for s in sources}

        datasets_to_process = {
            "PatternNet": (dataset_pn, pn_lab2cname), "UcMerced": (dataset_uc, uc_lab2cname),
            "EuroSAT": (dataset_euro, euro_lab2cname), "Mlrs": (dataset_mlrs, mlrs_lab2cname),
            "Milaid": (dataset_milaid, milaid_lab2cname),
        }
        for name, (dataset, lab2cname) in datasets_to_process.items():
            #print(f"  Applying merges to {name}...") # Less verbose
            update_lab2cname_for_merge(lab2cname, source_to_target_map)
            apply_merge_to_datums(dataset.train_x, source_to_target_map)
            apply_merge_to_datums(dataset.val, source_to_target_map)
            apply_merge_to_datums(dataset.test, source_to_target_map)
        print("Class merging complete.")

        # -- 4) Form global list of classes --
        print("Creating global class list...")
        all_updated_cnames = set()
        for _, (_, lab2cname) in datasets_to_process.items():
             all_updated_cnames.update(cname.lower() for cname in lab2cname.values())
        global_list = sorted(list(all_updated_cnames))
        global_num_classes = len(global_list)
        print(f"[INFO] Unified #classes after merging = {global_num_classes}")

        # -- 5) Build name->gid mapping --
        name2gid = {cname: i for i, cname in enumerate(global_list)}

        # -- 6) Set final self.lab2cname {global_id -> classname} --
        self.lab2cname = {i: cname for i, cname in enumerate(global_list)}
        # print("Global ID to Classname mapping (self.lab2cname):", self.lab2cname) # Optional detailed print

        # --- Sanity Check: Verify all classnames derived from updated dicts exist in global list ---
        print("Performing sanity check on class names...")
        error_found = False
        # Make the updated lab2cname dicts accessible for the check
        updated_lab2cnames = {
            "PatternNet": pn_lab2cname,
            "UcMerced": uc_lab2cname,
            "EuroSAT": euro_lab2cname,
            "Mlrs": mlrs_lab2cname,
            "Milaid": milaid_lab2cname,
        }

        # Iterate through the original datasets again for the check
        datasets_for_check = {
            "PatternNet": dataset_pn, "UcMerced": dataset_uc, "EuroSAT": dataset_euro,
            "Mlrs": dataset_mlrs, "Milaid": dataset_milaid
        }

        for dataset_name, dataset in datasets_for_check.items():
            current_lab2cname = updated_lab2cnames[dataset_name] # Get the relevant updated dict
            for split_name, split_data in [('train_x', dataset.train_x), ('val', dataset.val), ('test', dataset.test)]:
                if not split_data: continue
                for datum in split_data:
                    # Get the classname by looking up the original label in the *updated* dictionary
                    original_local_label = datum.label
                    correct_classname_from_dict = current_lab2cname.get(original_local_label)

                    if not correct_classname_from_dict or not isinstance(correct_classname_from_dict, str):
                        # This indicates an issue with the label itself or the dictionary update
                        print(f"ERROR: SanityCheck - Cannot find valid classname for original label {original_local_label} "
                            f"in {dataset_name} {split_name} updated lab2cname dict. Path: {datum.impath}")
                        error_found = True
                        continue # Skip to next datum

                    correct_classname_lower = correct_classname_from_dict.lower()

                    # Now check if this *correctly derived* classname exists in the global map
                    if correct_classname_lower not in name2gid:
                        print(f"ERROR: SanityCheck - Classname '{correct_classname_from_dict}' (derived from label {original_local_label}) "
                            f"in {dataset_name} {split_name} (path: ...{datum.impath[-50:]}) "
                            f"resolved to '{correct_classname_lower}' which is NOT FOUND in the final global name map (name2gid)!")
                        # Also print the datum's potentially stale classname for comparison
                        print(f"       (Datum's current classname attribute: '{getattr(datum, 'classname', 'MISSING')}')")
                        error_found = True

        if not error_found:
            print("Sanity check passed: All class names derived from updated dictionaries map to the global list.")
        else:
            # Add more debug info before raising error
            print("\nDebug Info: Final Global List (name2gid keys):")
            print(list(name2gid.keys()))
            # You could also print the problematic lab2cname dict here
            # print(f"\nProblematic lab2cname for {dataset_name}: {current_lab2cname}")
            raise ValueError("Classname mapping inconsistency found during sanity check after merging. Check renaming and merge logic.")
                # --- End Sanity Check ---

        # -- 7) Remap local labels -> global IDs --
        print("Remapping local labels to global IDs...")
        def remap(data_list, local_lab2cname_updated):
            if not data_list: return
            for idx, item in enumerate(data_list):
                original_local_label = item.label
                cname_from_dict = local_lab2cname_updated.get(original_local_label) # Safe lookup

                if cname_from_dict and isinstance(cname_from_dict, str):
                    cname_lower = cname_from_dict.lower()
                    if cname_lower in name2gid:
                         gid = name2gid[cname_lower]
                         new_datum_args = {
                             "impath": item.impath,
                             "label": gid,
                             "classname": self.lab2cname[gid]
                         }
                         for attr in ['domain', 'caption']: # Preserve optional attributes
                             if hasattr(item, attr):
                                 new_datum_args[attr] = getattr(item, attr)
                         data_list[idx] = Datum(**new_datum_args)
                    else:
                         raise ValueError(f"Cannot find global ID for class '{cname_lower}' derived from "
                                          f"local label {original_local_label}. Path: {item.impath}")
                else:
                     raise ValueError(f"Could not find valid classname for original local label {original_local_label} "
                                      f"in updated lab2cname dict. Path: {item.impath}")

        remap(dataset_pn.train_x, pn_lab2cname); remap(dataset_pn.val, pn_lab2cname); remap(dataset_pn.test, pn_lab2cname)
        remap(dataset_uc.train_x, uc_lab2cname); remap(dataset_uc.val, uc_lab2cname); remap(dataset_uc.test, uc_lab2cname)
        remap(dataset_euro.train_x, euro_lab2cname); remap(dataset_euro.val, euro_lab2cname); remap(dataset_euro.test, euro_lab2cname)
        remap(dataset_mlrs.train_x, mlrs_lab2cname); remap(dataset_mlrs.val, mlrs_lab2cname); remap(dataset_mlrs.test, mlrs_lab2cname)
        remap(dataset_milaid.train_x, milaid_lab2cname); remap(dataset_milaid.val, milaid_lab2cname); remap(dataset_milaid.test, milaid_lab2cname)
        print("Label remapping complete.")

        # -- 8) Overwrite cfg.MODEL.NUM_CLASSES --
        self.cfg.defrost()
        self.cfg.MODEL.NUM_CLASSES = global_num_classes
        self.cfg.freeze()
        print(f"Set cfg.MODEL.NUM_CLASSES = {global_num_classes}")

        # -- 9) Create ClientDataManager instances --
        print("Creating ClientDataManagers...")
        dm_client_0 = ClientDataManager(train_x=dataset_pn.train_x, val=dataset_pn.val, test=dataset_pn.test, cfg=self.cfg)
        dm_client_1 = ClientDataManager(train_x=dataset_uc.train_x, val=dataset_uc.val, test=dataset_uc.test, cfg=self.cfg)
        dm_client_2 = ClientDataManager(train_x=dataset_euro.train_x, val=dataset_euro.val, test=dataset_euro.test, cfg=self.cfg)
        dm_client_3 = ClientDataManager(train_x=dataset_mlrs.train_x, val=dataset_mlrs.val, test=dataset_mlrs.test, cfg=self.cfg)
        dm_client_4 = ClientDataManager(train_x=dataset_milaid.train_x, val=dataset_milaid.val, test=dataset_milaid.test, cfg=self.cfg)
        self.client_data_managers = [dm_client_0, dm_client_1, dm_client_2, dm_client_3, dm_client_4]

        # Final setup for TrainerX
        self.train_loader_x = None; self.val_loader = None; self.test_loader = None; self.dm = None
        print("build_data_loader finished successfully.")


    def create_unified_test_dataloader(self):
        """Creates a DataLoader for the combined test sets of all clients."""
        print("\nCreating unified test dataloader...")
        test_sets = [ getattr(ds, 'test', []) for ds in [self.dataset_pn, self.dataset_uc, self.dataset_euro, self.dataset_mlrs, self.dataset_milaid]]
        unified_test = [item for sublist in test_sets if sublist for item in sublist] # Combine lists safely

        if not unified_test:
            print("Warning: Unified test set is empty!")
            return None # Return None if empty

        print(f"Total unified test samples: {len(unified_test)}")

        # Use transforms from client 0 (assuming TEST transforms are consistent)
        test_transform = self.client_data_managers[0].tfm_test
        if test_transform is None:
            print("Warning: Using default/None test transforms.")

        # Create a Dataset compatible with DataLoader
        class ListDataset(torch.utils.data.Dataset):
            def __init__(self, data_list, transform=None):
                self.data_list = data_list
                self.transform = transform

            def __len__(self):
                return len(self.data_list)

            def __getitem__(self, idx):
                item = self.data_list[idx]
                img_path = item.impath
                label = item.label
                try:
                    img = Image.open(img_path).convert("RGB")
                    if self.transform:
                        img = self.transform(img)
                    # Return in a dictionary format expected by _evaluate_model_on_dataloader
                    return {'img': img, 'label': label}
                except Exception as e:
                     print(f"Error loading image {img_path} for unified test set: {e}. Returning None.")
                     # DataLoader default collate should handle None if necessary, or skip in eval loop
                     # Safer to return dummy data? Let's try skipping in eval loop first.
                     # Return dummy data of correct types (might cause issues downstream)
                     # return {'img': torch.zeros(3, self.cfg.INPUT.SIZE[0], self.cfg.INPUT.SIZE[1]), 'label': -1}
                     raise RuntimeError(f"Failed to load image: {img_path}") from e


        unified_dataset = ListDataset(unified_test, transform=test_transform)

        # Create the DataLoader manually
        unified_loader = DataLoader(
            unified_dataset,
            batch_size=self.cfg.DATALOADER.TEST.BATCH_SIZE, # Use test batch size
            shuffle=False, # No shuffling for evaluation
            num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            pin_memory=(torch.cuda.is_available() and self.cfg.USE_CUDA),
            drop_last=False
        )
        print(f"Unified test dataloader created with {len(unified_test)} samples.")
        return unified_loader


    def create_unified_train_dataloader(self):
        """
        Creates a DataLoader for the combined training sets of all clients.
        Uses training transforms from client 0. For EVALUATION purposes (no shuffle).
        """
        print("\nCreating unified training dataloader (for evaluation)...")
        if not self.client_data_managers:
            print("ERROR: No client data managers available.")
            return None

        unified_train_data = []
        total_samples = 0
        for i, dm in enumerate(self.client_data_managers):
            train_list = getattr(dm, 'train_x_list', [])
            if train_list:
                unified_train_data.extend(train_list)
                count = len(train_list)
                total_samples += count
                # print(f"  Added {count} training samples from client {i}")
            #else: print(f"  Client {i} has no training samples.")

        if not unified_train_data:
            print("ERROR: No training data found across all clients.")
            return None

        print(f"Total combined training samples: {total_samples}")

        # Use TEST transforms for evaluation consistency, even on train data
        eval_transform = self.client_data_managers[0].tfm_test
        if eval_transform is None:
            print("Warning: Using default/None transforms for training set evaluation.")

        # Create a Dataset compatible with DataLoader (same as for test)
        class ListDataset(torch.utils.data.Dataset):
            def __init__(self, data_list, transform=None):
                self.data_list = data_list
                self.transform = transform
            def __len__(self): return len(self.data_list)
            def __getitem__(self, idx):
                item = self.data_list[idx]
                img_path = item.impath; label = item.label
                try:
                    img = Image.open(img_path).convert("RGB")
                    if self.transform: img = self.transform(img)
                    return {'img': img, 'label': label}
                except Exception as e:
                     print(f"Error loading image {img_path} for unified train eval: {e}. Raising error.")
                     raise RuntimeError(f"Failed to load image: {img_path}") from e

        unified_dataset = ListDataset(unified_train_data, transform=eval_transform)

        # Create the DataLoader manually for evaluation
        unified_loader = DataLoader(
            unified_dataset,
            batch_size=self.cfg.DATALOADER.TEST.BATCH_SIZE, # Use test batch size for eval
            shuffle=False, # No shuffling for evaluation
            num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            pin_memory=(torch.cuda.is_available() and self.cfg.USE_CUDA),
            drop_last=False
        )
        print(f"Unified training dataloader (for evaluation) created with {len(unified_train_data)} samples.")
        return unified_loader

    # --------------------------------------------------------------------------
    # NEW: Generic Evaluation Method
    # --------------------------------------------------------------------------
    def _evaluate_model_on_dataloader(self, model, dataloader, dataset_name="Dataset", print_details=True):
        """
        Evaluates a given model on a provided dataloader.

        Args:
            model: The model to evaluate.
            dataloader: The DataLoader containing the evaluation data.
            dataset_name: A string name for the dataset (for printing).
            print_details: If True, prints per-class accuracy and misclassification details.

        Returns:
            A dictionary containing evaluation metrics.
        """
        model.eval()
        device = self.device
        num_classes = self.cfg.MODEL.NUM_CLASSES

        total_loss = 0.0 # If loss calculation is needed
        total_correct = 0
        total_samples = 0

        # For per-class stats
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        misclassifications = defaultdict(lambda: defaultdict(int)) # Store true_label -> {pred_label: count}

        if dataloader is None:
            print(f"Error: DataLoader for {dataset_name} is None. Cannot evaluate.")
            return None

        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating on {dataset_name}", leave=False)):
                # Handle potential errors from dataset __getitem__ returning None or exceptions
                if batch is None:
                     print(f"Warning: Skipping None batch encountered in {dataset_name} dataloader at index {batch_idx}.")
                     continue

                try:
                    if isinstance(batch, dict) and 'img' in batch and 'label' in batch:
                        images, labels = batch['img'].to(device), batch['label'].to(device)
                        # Skip batch if labels indicate an error (e.g., -1 from dataset)
                        if torch.any(labels < 0):
                            print(f"Warning: Skipping batch {batch_idx} due to invalid labels (< 0).")
                            continue
                    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
                         images, labels = batch[0].to(device), batch[1].to(device)
                         if torch.any(labels < 0):
                            print(f"Warning: Skipping batch {batch_idx} due to invalid labels (< 0).")
                            continue
                    else:
                         print(f"ERROR: Unexpected batch format in {dataset_name} loader: {type(batch)}")
                         continue # Skip this batch

                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)

                    # Update overall and per-class stats
                    valid_indices = (labels >= 0) & (labels < num_classes) # Ensure labels are valid
                    valid_labels = labels[valid_indices]
                    valid_predicted = predicted[valid_indices]

                    total_samples += valid_labels.size(0) # Count only valid samples processed
                    total_correct += (valid_predicted == valid_labels).sum().item()

                    for i in range(valid_labels.size(0)): # Iterate only over valid samples
                        true_label = valid_labels[i].item()
                        pred_label = valid_predicted[i].item()

                        class_total[true_label] += 1
                        if true_label == pred_label:
                            class_correct[true_label] += 1
                        else:
                            # Check predicted label validity before recording misclassification
                             if 0 <= pred_label < num_classes:
                                misclassifications[true_label][pred_label] += 1

                except Exception as e:
                     print(f"ERROR during evaluation forward pass in {dataset_name} (batch {batch_idx}): {e}")
                     import traceback
                     traceback.print_exc()
                     continue # Skip batch on error

        # Calculate overall results
        overall_accuracy = (100.0 * total_correct / total_samples) if total_samples > 0 else 0.0

        print(f"\n=== Evaluation Results: {dataset_name} ===")
        print(f"Overall Accuracy: {total_correct} / {total_samples} ({overall_accuracy:.2f}%)")

        # Print per-class details if requested
        if print_details:
            print("\n--- Per-Class Accuracy & Top Misclassifications ---")
            class_accuracies = {}
            for class_id in range(num_classes):
                class_name = self.lab2cname.get(class_id, f"Unknown Class {class_id}")
                correct = class_correct[class_id]
                total = class_total[class_id]

                if total > 0:
                    accuracy = 100.0 * correct / total
                    print(f"Class {class_id:03d} ({class_name:<25}): {correct:4d} / {total:4d} ({accuracy:6.2f}%) Correct")
                    class_accuracies[class_name] = accuracy

                    # Print top misclassifications for this class
                    if class_id in misclassifications:
                        sorted_errors = sorted(misclassifications[class_id].items(), key=lambda item: item[1], reverse=True)
                        for i, (predicted_as_id, count) in enumerate(sorted_errors):
                            if i >= 3: break # Limit printed errors
                            predicted_as_name = self.lab2cname.get(predicted_as_id, f"Unknown Class {predicted_as_id}")
                            print(f"      -> Predicted as '{predicted_as_name}' ({predicted_as_id:03d}): {count} times")
                else:
                    # Optionally print classes with no samples
                    # print(f"Class {class_id:03d} ({class_name:<25}):    0 /    0 (  N/A  %) Correct (No samples)")
                    pass

            print("--- End Per-Class Stats ---")
            # Log per-class accuracies to wandb if desired
            # wandb.log({f"per_class_acc/{k}": v for k, v in class_accuracies.items()})


        return {
            "accuracy": overall_accuracy,
            "total_samples": total_samples,
            "class_correct": class_correct, # Raw counts
            "class_total": class_total,       # Raw counts
            "misclassifications": misclassifications # Detailed errors
        }
    # --------------------------------------------------------------------------

    ###################################################
    # B) Build local trainers (DualPrompt)
    ###################################################
    def build_model(self):
        self.clients = []
        if not self.lab2cname:
             raise ValueError("self.lab2cname is not populated. Ensure build_data_loader runs first.")
        global_classnames = list(self.lab2cname.values())
        print(f"Building models for {self.num_clients} clients with {len(global_classnames)} global classes.")

        for i, dm in enumerate(self.client_data_managers):
            print(f"  Building model for client {i}...")
            local_trainer = DualPrompt(self.cfg, client_id=i, classnames=global_classnames, _clip_model=self._clip_model)
            local_trainer.dm = dm
            local_trainer.build_model()
            self.clients.append(local_trainer)

        if self.global_weights is None:
             print("Initializing global weights from client 0.")
             self.global_weights = copy.deepcopy(self.clients[0].model.state_dict())
        else:
             print("Global weights already loaded, skipping initialization from client 0.")
        print("Model building complete.")

    ###################################################
    # C) Federated training loop
    ###################################################
    def train(self):
        # Start W&B run if configured
        
             #wandb.init(project=self.cfg.WANDB.PROJECT, entity=self.cfg.WANDB.ENTITY, config=self.cfg, name=self.cfg.WANDB.RUN_NAME)
        wandb.watch(self.clients[0].model) # Watch one model

        for round_idx in range(self.num_rounds):
            print(f"\n--- Federated Round {round_idx+1}/{self.num_rounds} ---")

            # 1) Broadcast global weights
            if self.check_weights_valid(self.global_weights):
                try:
                    self.broadcast_weights(self.global_weights)
                except ValueError as e: # Catch broadcast error
                    print(f"ERROR during broadcast: {e}. Skipping round.")
                    self.nan_stats["skipped_rounds"] += 1
                    continue
            else:
                print("WARNING: Invalid global weights detected before broadcast! Skipping round.")
                self.nan_stats["skipped_rounds"] += 1
                continue

            local_state_dicts = []
            valid_clients_indices = []
            valid_clients_trainers = []
            round_losses = []

            # 2) Local Training
            for i, trainer in enumerate(self.clients):
                print(f"[Client {i}] starting local training ({self.local_epochs} epochs)...")
                trainer.epoch = round_idx * self.local_epochs
                trainer.max_epoch = (round_idx + 1) * self.local_epochs
                last_epoch_loss = None

                try:
                    for ep in range(trainer.epoch, trainer.max_epoch):
                        epoch_res = trainer.run_epoch(ep)
                        if epoch_res and "avg_loss" in epoch_res: last_epoch_loss = epoch_res["avg_loss"]
                    print(f"[Client {i}] finished local training.")
                    if last_epoch_loss is not None:
                        round_losses.append(last_epoch_loss)
                        if wandb.run: wandb.log({"round": round_idx, f"client_{i}_final_local_loss": last_epoch_loss}, step=round_idx)

                    w = trainer.model.state_dict()
                    if self.check_weights_valid(w):
                        local_state_dicts.append(copy.deepcopy(w))
                        valid_clients_indices.append(i)
                        valid_clients_trainers.append(trainer)
                    else:
                        print(f"WARNING: Client {i} produced invalid weights, discarding.")
                        self.nan_stats["failed_clients"].append((round_idx, i, "Invalid weights after train"))

                except Exception as e: # Catch broader exceptions during training
                    print(f"ERROR: Client {i} failed during local training: {str(e)}")
                    import traceback; traceback.print_exc()
                    self.nan_stats["failed_clients"].append((round_idx, i, str(e)))
                    continue

            # 3) Log Average Loss
            if round_losses:
                avg_loss_this_round = sum(round_losses) / len(round_losses)
                print(f"[Round {round_idx+1}] Average final local epoch loss across {len(round_losses)} clients = {avg_loss_this_round:.4f}")
                if wandb.run: wandb.log({"round": round_idx, "avg_final_local_loss": avg_loss_this_round}, step=round_idx)
            else: print(f"[Round {round_idx+1}] No clients reported successful training losses.")

            # 4) Aggregation
            if valid_clients_trainers:
                print(f"Aggregating weights from {len(valid_clients_trainers)} clients: {valid_clients_indices}")
                new_global_weights = self.safe_average_weights(local_state_dicts, valid_clients_trainers)
                if self.check_weights_valid(new_global_weights):
                    self.global_weights = new_global_weights
                    self.nan_stats['total_updates'] += 1
                    print("Global weights updated successfully.")
                else:
                    print("WARNING: Aggregated global weights are invalid! Reverting.")
                    self.nan_stats['skipped_rounds'] += 1
            else:
                print("WARNING: No clients produced valid weights! Skipping aggregation.")
                self.nan_stats['skipped_rounds'] += 1

            # 5) Periodic Evaluation (on Unified Test Set)
            if (round_idx + 1) % 1 == 0 or (round_idx + 1) == self.num_rounds:
                 if self.check_weights_valid(self.global_weights):
                     print(f"\n--- Evaluating Global Model at Round {round_idx+1} ---")
                     # Ensure clients have latest weights for eval helpers if needed elsewhere
                     self.broadcast_weights(self.global_weights)

                     # Evaluate on Unified Test Set
                     try:
                         unified_test_loader = self.create_unified_test_dataloader()
                         if unified_test_loader:
                             # Use client 0's model structure for evaluation
                             # The _evaluate... method handles printing details
                             unified_test_res = self._evaluate_model_on_dataloader(
                                 self.clients[0].model,
                                 unified_test_loader,
                                 dataset_name="Unified Test Set (Round {})".format(round_idx+1),
                                 print_details=True # <<< Enable detailed printing
                             )
                             if unified_test_res and "accuracy" in unified_test_res:
                                 unified_acc = unified_test_res["accuracy"]
                                 print(f"=== [Round {round_idx+1}] Unified Test Accuracy = {unified_acc:.2f}% ===")
                                 if wandb.run: wandb.log({"round": round_idx, "eval_unified_test_accuracy": unified_acc}, step=round_idx)
                         else:
                              print("  Could not create unified test loader for evaluation.")
                     except Exception as e:
                          print(f"ERROR during unified test evaluation at round {round_idx+1}: {e}")
                          import traceback; traceback.print_exc()

                     # Optional: Evaluate on individual client test sets too
                     # self.test_on_all_clients(round_idx) # Pass round for logging step

                 else:
                     print(f"WARNING: Global weights invalid at round {round_idx+1}, skipping evaluation.")

            # 6) Periodic Model Saving
            if (round_idx + 1) % 5 ==0 :
                if self.check_weights_valid(self.global_weights):
                    print(f"Saving global model checkpoint at round {round_idx+1}...")
                    self.save_model(epoch=(round_idx + 1)) # Use round number as epoch
                else:
                    print(f"Skipping checkpoint at round {round_idx+1} due to invalid weights.")


        # End Training Loop
        print("\nFederated training finished.")
        self.finalize_training()
        if wandb.run: wandb.finish()


    # This is now primarily for evaluating on the UNIFIED test set
    def test_on_unified_dataset(self, test_loader):
        """
        Evaluates the current global model (on client 0) on the unified test dataset.
        Calls the internal helper _evaluate_model_on_dataloader.
        """
        if not self.clients:
             print("ERROR: No clients available for unified testing.")
             return None
        if not test_loader:
            print("ERROR: Unified test loader is invalid or empty.")
            return None
        if not self.check_weights_valid(self.global_weights):
            print("ERROR: Global weights are invalid. Cannot perform unified test.")
            # Optionally load last known good weights or return None
            return None

        # Ensure client 0 has the weights to be tested
        try:
            self.clients[0].model.load_state_dict(self.global_weights, strict=False)
        except Exception as e:
             print(f"Error loading global weights onto client 0 for testing: {e}")
             return None

        model_to_test = self.clients[0].model
        print(f"\nEvaluating Global Model on Unified Test Set...")

        # Call the generic evaluation method with detailed printing enabled
        results = self._evaluate_model_on_dataloader(
            model_to_test,
            test_loader,
            dataset_name="Unified Test Set",
            print_details=True # <<< Ensure details are printed
        )
        return results


    # ... (safe_average_weights, _calculate_diversity, check_weights_valid) ...
    # ... (compute_file_hash, compute_state_dict_hash, broadcast_weights) ...

    # Keep safe_average_weights, _calculate_diversity, check_weights_valid,
    # compute_file_hash, compute_state_dict_hash, broadcast_weights as they were,
    # ensuring check_weights_valid is robust.

    def safe_average_weights(self, local_dicts, valid_clients_trainers):
        if not local_dicts:
            print("WARNING: No local dictionaries provided for averaging. Returning previous global weights.")
            # Ensure self.global_weights exists and is valid before returning
            if self.global_weights is None or not self.check_weights_valid(self.global_weights):
                 print("ERROR: Cannot average, no valid local dicts and no valid previous global weights!")
                 raise ValueError("Averaging failed: No valid weights available.")
            return self.global_weights

        diversity_scores = [self._calculate_diversity(client) for client in valid_clients_trainers]
        total_score = sum(diversity_scores)

        weights = []
        if total_score > 1e-9:
            weights = [score / total_score for score in diversity_scores]
        else:
            print("Diversity scores sum to near zero, falling back to uniform averaging.")
            num_clients = len(local_dicts)
            weights = [1.0 / num_clients] * num_clients if num_clients > 0 else []

        #print(f"Averaging weights with diversity weights: {[f'{w:.3f}' for w in weights]}")

        avg_state = OrderedDict()
        first_dict_keys = local_dicts[0].keys()

        for key in first_dict_keys:
            valid_tensors_for_key = []
            valid_weights_for_key = []

            # Collect valid tensors and corresponding weights for the current key
            for sd, weight in zip(local_dicts, weights):
                if key in sd:
                    tensor = sd[key].float() # Use float32 for stability
                    if not torch.isnan(tensor).any() and not torch.isinf(tensor).any():
                        valid_tensors_for_key.append(tensor)
                        valid_weights_for_key.append(weight)
                    #else: print(f"Warning: NaN/Inf in tensor '{key}' from a client, skipping.") # Less verbose
                #else: print(f"Warning: Key '{key}' missing in a client dict, skipping.") # Less verbose

            if not valid_tensors_for_key:
                #print(f"Warning: No valid tensors to average for key '{key}'.") # Less verbose
                # Try to keep the old global value if possible and valid
                if self.global_weights and key in self.global_weights and self.check_weights_valid({key: self.global_weights[key]}):
                    avg_state[key] = self.global_weights[key].clone()
                    #print(f"  Kept previous valid global value for key '{key}'.")
                else:
                    print(f"ERROR: Cannot average key '{key}' and no valid previous value exists!")
                    # This might require stopping or more sophisticated handling
                continue

            # Renormalize weights if some tensors were skipped
            current_total_weight = sum(valid_weights_for_key)
            if current_total_weight < 1e-9: # All weights zero or skipped
                 # Fallback: simple average of the valid tensors found
                 if valid_tensors_for_key:
                     stacked = torch.stack(valid_tensors_for_key, dim=0)
                     averaged_tensor = torch.mean(stacked, dim=0)
                 else: # Should not happen if we checked valid_tensors_for_key earlier
                     print(f"ERROR: Logic error averaging key {key}, no valid tensors but weight sum is zero.")
                     continue
            else:
                normalized_weights = [w / current_total_weight for w in valid_weights_for_key]
                # Apply weights
                weighted_tensors = [tensor * weight for tensor, weight in zip(valid_tensors_for_key, normalized_weights)]
                # Stack and sum
                try:
                    stacked = torch.stack(weighted_tensors, dim=0)
                    averaged_tensor = torch.sum(stacked, dim=0)
                except RuntimeError as e:
                     print(f"ERROR during stacking/summing for key '{key}': {e}. Trying to keep previous value.")
                     if self.global_weights and key in self.global_weights and self.check_weights_valid({key: self.global_weights[key]}):
                          avg_state[key] = self.global_weights[key].clone()
                     continue # Skip to next key

            # Final check on the averaged tensor for this key
            if torch.isnan(averaged_tensor).any() or torch.isinf(averaged_tensor).any():
                 print(f"ERROR: NaN/Inf detected in aggregated tensor for key '{key}'! Trying to keep previous value.")
                 if self.global_weights and key in self.global_weights and self.check_weights_valid({key: self.global_weights[key]}):
                    avg_state[key] = self.global_weights[key].clone()
                 # else: Raise error?
            else:
                # Assign the valid averaged tensor, casting to original dtype if possible
                target_dtype = self.global_weights[key].dtype if self.global_weights and key in self.global_weights else averaged_tensor.dtype
                avg_state[key] = averaged_tensor.to(target_dtype)

        # Final check on the whole averaged state dict
        if not self.check_weights_valid(avg_state):
             print("ERROR: The final averaged state dictionary contains invalid values. Averaging failed. Returning previous valid global weights.")
             if self.global_weights is None or not self.check_weights_valid(self.global_weights):
                  raise ValueError("Averaging failed: No valid weights available.")
             return self.global_weights # Return the last known good weights

        return avg_state

    def _calculate_diversity(self, client):
        """Calculate normalized entropy diversity score for a client based on training data."""
        if not hasattr(client, 'dm') or not hasattr(client.dm, 'train_x_list') or not client.dm.train_x_list:
            #print("Warning: Client missing training data for diversity calculation. Returning low diversity score.")
            return 0.0
        class_counts = Counter([d.classname for d in client.dm.train_x_list])
        if not class_counts: return 0.0
        counts = torch.tensor(list(class_counts.values()), dtype=torch.float32)
        total_samples = counts.sum()
        if total_samples == 0: return 0.0
        probabilities = counts / total_samples
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-12))
        num_classes_present = len(class_counts)
        if num_classes_present <= 1: return 0.0
        max_entropy = torch.log(torch.tensor(num_classes_present, dtype=torch.float32))
        normalized_entropy = entropy / (max_entropy + 1e-12)
        diversity_score = torch.clamp(normalized_entropy, 0.0, 1.0).item()
        return diversity_score

    def check_weights_valid(self, state_dict):
        """Checks if a state_dict contains any None, NaN or Inf values."""
        if state_dict is None:
            print("ERROR: State dictionary is None.")
            return False
        for name, param in state_dict.items():
            if param is None:
                print(f"ERROR: Parameter '{name}' is None.")
                return False
            # Check if param is a tensor before calling torch functions
            if isinstance(param, torch.Tensor):
                if torch.isnan(param).any():
                    print(f"ERROR: NaN detected in parameter '{name}'.")
                    return False
                if torch.isinf(param).any():
                    print(f"ERROR: Inf detected in parameter '{name}'.")
                    return False
            # Optionally handle non-tensor parameters if they exist, otherwise assume valid
            # else: print(f"Warning: Non-tensor parameter '{name}' found.")
        return True

    def compute_file_hash(self, path):
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        try:
            with open(path, 'rb') as f:
                while chunk := f.read(8192): sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e: print(f"ERROR hashing file {path}: {e}"); return None

    def compute_state_dict_hash(self, state_dict):
        """Compute SHA-256 hash of the state_dict."""
        if state_dict is None: return None
        sha256 = hashlib.sha256()
        try:
            for key in sorted(state_dict.keys()):
                tensor = state_dict[key]
                sha256.update(key.encode('utf-8'))
                if isinstance(tensor, torch.Tensor):
                     sha256.update(tensor.cpu().numpy().tobytes())
                elif isinstance(tensor, (int, float, str, bool)): # Handle simple types
                     sha256.update(str(tensor).encode('utf-8'))
                # Add handling for other types if necessary
            return sha256.hexdigest()
        except Exception as e: print(f"ERROR computing state dict hash: {e}"); return None

    def broadcast_weights(self, global_sd):
        """Broadcast global state_dict to each client model. Resets optimizer and scheduler."""
        if not self.check_weights_valid(global_sd):
             raise ValueError("Cannot broadcast invalid weights.") # Raise error immediately

        state_hash = self.compute_state_dict_hash(global_sd)
        # print(f"Broadcasting global weights (hash: {state_hash[:8]}...) to {len(self.clients)} clients.") # Less verbose

        for i, client_trainer in enumerate(self.clients):
            try:
                load_result = client_trainer.model.load_state_dict(global_sd, strict=False)
                if load_result.missing_keys: print(f"Warning: Client {i} missing keys: {load_result.missing_keys}")
                if load_result.unexpected_keys: print(f"Warning: Client {i} unexpected keys: {load_result.unexpected_keys}")

                # Reset optimizer state
                # Simple reset: client_trainer.optim.state = defaultdict(dict)
                # More robust reset: Rebuild optimizer (might be needed if e.g. layer freezing changes)
                client_trainer.optim = build_optimizer(client_trainer.model, client_trainer.cfg.OPTIM)

                # Rebuild the learning rate scheduler
                client_trainer.sched = build_lr_scheduler(client_trainer.optim, client_trainer.cfg.OPTIM)
                if hasattr(client_trainer, "epoch") and hasattr(client_trainer.sched, 'last_epoch'):
                    # Sync scheduler's epoch. Dassl schedulers step *after* optimizer step,
                    # so last_epoch should be the completed epoch number.
                    # If trainer.epoch is the *next* epoch to run, last_epoch is trainer.epoch - 1.
                     client_trainer.sched.last_epoch = max(-1, client_trainer.epoch - 1) # Ensure >= -1

            except Exception as e:
                print(f"ERROR broadcasting weights to client {i}: {e}")
                raise RuntimeError(f"Broadcast failed for client {i}") from e # Propagate error


    def finalize_training(self):
        """Summarize training results and evaluate the final model."""
        print("\n======== Training Summary ========")
        # ... (Print round/failure stats as before) ...
        print(f"Total Rounds Attempted: {self.num_rounds}")
        print(f"Successful Global Updates: {self.nan_stats['total_updates']}")
        print(f"Skipped Rounds: {self.nan_stats['skipped_rounds']}")
        if self.nan_stats['failed_clients']:
             print(f"Client Failures Recorded ({len(self.nan_stats['failed_clients'])} instances):")
             # ... (Summarize failures) ...
        else: print("\nNo client failures recorded.")


        # Evaluate final global model
        print("\n======== Evaluating Final Global Model ========")
        if self.check_weights_valid(self.global_weights):
            try:
                self.broadcast_weights(self.global_weights) # Ensure clients have final weights
            except (ValueError, RuntimeError) as e:
                 print(f"ERROR: Could not broadcast final weights for evaluation: {e}. Aborting final evaluation.")
                 return # Stop if broadcast fails

            # Evaluate on Unified Test Data (with details)
            try:
                print("\n--- Final Evaluation on Unified Test Set ---")
                unified_test_loader = self.create_unified_test_dataloader()
                if unified_test_loader:
                    final_unified_results = self._evaluate_model_on_dataloader(
                        self.clients[0].model,
                        unified_test_loader,
                        dataset_name="Unified Test Set (Final)",
                        print_details=True # <<< Print details
                    )
                    if final_unified_results and "accuracy" in final_unified_results:
                         print(f"\n>>> Final Unified Test Accuracy: {final_unified_results['accuracy']:.2f}% <<<")
                         if wandb.run: wandb.log({"final_eval/unified_test_accuracy": final_unified_results['accuracy']})
                else:
                    print("  Could not create unified test loader for final evaluation.")
            except Exception as e:
                 print(f"  ERROR during final unified test evaluation: {e}")
                 import traceback; traceback.print_exc()

            # Evaluate on Combined Training Data (with details)
            try:
                print("\n--- Final Evaluation on Combined Training Set ---")
                unified_train_loader = self.create_unified_train_dataloader()
                if unified_train_loader:
                    final_train_results = self._evaluate_model_on_dataloader(
                        self.clients[0].model,
                        unified_train_loader,
                        dataset_name="Combined Training Set (Final)",
                        print_details=True # <<< Print details for train set too
                    )
                    if final_train_results and "accuracy" in final_train_results:
                        print(f"\n>>> Final Combined Training Accuracy: {final_train_results['accuracy']:.2f}% <<<")
                        if wandb.run: wandb.log({"final_eval/combined_train_accuracy": final_train_results['accuracy']})
                else:
                    print("  Could not create unified training loader for final evaluation.")
            except Exception as e:
                 print(f"  ERROR during final training set evaluation: {e}")
                 import traceback; traceback.print_exc()

            # Optional: Evaluate on Client 0's test set
            # try:
            #      print("\n--- Final Evaluation on Client 0 Test Set ---")
            #      final_result_c0 = self.clients[0].test() # Assumes client test method exists
            #      if final_result_c0 and "accuracy" in final_result_c0:
            #           print(f"  Final Test Accuracy (Client 0): {final_result_c0['accuracy']:.2f}%")
            #           if wandb.run: wandb.log({"final_eval/client_0_test_accuracy": final_result_c0['accuracy']})
            # except Exception as e: print(f"  ERROR during final client 0 test evaluation: {e}")


            # Save the final model
            print("\nSaving final global model...")
            self.before_save()
            self.save_model(epoch=self.num_rounds)

        else:
            print("\nFinal global weights are invalid. Cannot perform final evaluation or save.")
        print("===================================")


    def before_save(self):
        """Sync global weights to base model if needed by save_checkpoint."""
        if hasattr(self, '_models') and self._models:
             model_names = self.get_model_names()
             if model_names:
                primary_model_name = model_names[0]
                if primary_model_name in self._models:
                    #print(f"Syncing global_weights into self._models['{primary_model_name}'] before saving.") # Less verbose
                    if self.check_weights_valid(self.global_weights):
                        self._models[primary_model_name].load_state_dict(self.global_weights, strict=False)
                    else: print("Warning: Cannot sync invalid global_weights to self._models.")


    # ========================================================================
    # save_model / load_model - DO NOT CHANGE (as per user request)
    # ========================================================================
    def save_model(self, epoch=None, directory="", is_best=False, val_result=None):
        if not directory:
            directory = self.cfg.OUTPUT_DIR
        mkdir_if_missing(directory)

        subfolder = "DualPromptPromptLearner_Aggregator"
        target_dir = osp.join(directory, subfolder)
        mkdir_if_missing(target_dir)

        checkpoint = {
            "epoch": self.cfg.OPTIM.MAX_EPOCH,
            "state_dict": self.global_weights,
            "optimizer": None,
            "scheduler": None,
            "val_result": val_result,
            "cfg": self.cfg.dump()
        }
        # --- Keep these lines exactly as-is ---
        fpath=save_checkpoint(checkpoint, target_dir, is_best=is_best)
        if self.cfg.VERBOSE:
            print(f"Model saved to {target_dir}")
        # --- End of the original code we must NOT change ---

        # -------------------------------------------------------------------
        # ADD: Identify the file that Dassl just saved and log it as a W&B artifact
        # -------------------------------------------------------------------

        # By Dassl convention, providing a directory to save_checkpoint creates:
        #  - "model.pth.tar" inside target_dir
        #  - if is_best=True, also creates "model-best.pth.tar"
        #model_fpath = osp.join(target_dir, "model.pth.tar-2")
        model_fpath = fpath

        model_best_fpath = osp.join(target_dir, "model-best.pth.tar")

        # Decide which file to upload (if is_best, we try the "best" file first)
        if is_best and osp.exists(model_best_fpath):
            final_path = model_best_fpath
        else:
            final_path = model_fpath

        # Create an artifact for W&B
        artifact = wandb.Artifact(
            name="aggregator_checkpoint",
            type="model",
            metadata={
                "epoch": epoch,
                "is_best": is_best
            }
        )
        artifact.add_file(final_path)  # attach the file that actually exists
        wandb.log_artifact(artifact)

        if self.cfg.VERBOSE:
            print(f"W&B artifact logged from {final_path}")
        # --- End W&B Artifact Logging ---
    def load_model(self, directory, epoch=None, expected_file_hash=None):
        """Load aggregator weights with sanity checks."""
        if not directory:
            print("Skipping load_model, no pretrained path given")
            return

        subfolder = "DualPromptPromptLearner_Aggregator"
        model_file = f"model.pth.tar-{epoch}" if epoch else "model.pth.tar"
        path = osp.join(directory, subfolder, model_file)
        
        if not osp.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")

        # Verify file integrity
        file_hash = self.compute_file_hash(path)
        print(f"Model file SHA-256: {file_hash}")
        if expected_file_hash and file_hash != expected_file_hash:
            raise ValueError("Model file hash mismatch!")

        ckpt = load_checkpoint(path)
        state_dict = ckpt["state_dict"]
        loaded_epoch = ckpt.get("epoch", None)


        # Log keys before deletion
        print("Keys before deletion:", state_dict.keys())

        # Delete specific keys
        deleted_keys = []
        for key in ["prompt_learner.token_prefix", "prompt_learner.token_suffix"]:
            if key in state_dict:
                del state_dict[key]
                deleted_keys.append(key)
        print(f"Deleted keys: {deleted_keys}")

        # Verify deletion
        for key in deleted_keys:
            assert key not in state_dict, f"{key} was not deleted!"

        # Log keys after deletion
        print("Keys after deletion:", state_dict.keys())

        # Compute and log state dict hash
        state_hash = self.compute_state_dict_hash(state_dict)
        print(f"State dict hash: {state_hash}")

        self.global_weights = state_dict
        print(f"Loaded aggregator weights from '{path}' (epoch={loaded_epoch}).")

        # Enhanced validity check (include checks for deleted keys)
        if self.check_weights_valid(self.global_weights):
            # Broadcast hash for client verification
            self.broadcast_weights(self.global_weights)
            print(f"Broadcasted weights with hash: {state_hash}")
        else:
            print("Warning: Loaded global weights invalid! Skipping broadcast.")


    # ========================================================================
    # Other Methods (test_on_all_clients, debug methods)
    # ========================================================================
    def test_on_all_clients(self, current_round=None): # Add round for logging step
        """Test the current global model on all clients' local test sets."""
        if not self.clients: print("No clients initialized for testing."); return
        print("\n--- Testing Global Model on All Client Test Sets ---")
        if not self.check_weights_valid(self.global_weights):
             print("ERROR: Global weights are invalid. Cannot perform testing."); return

        try: # Ensure broadcast happens before testing loop
            self.broadcast_weights(self.global_weights)
        except (ValueError, RuntimeError) as e:
             print(f"ERROR: Could not broadcast weights for client testing: {e}. Aborting.")
             return

        all_results = {}
        log_data = {"round": current_round} if current_round is not None else {}
        for i, trainer in enumerate(self.clients):
            #print(f"\n--- Testing on Client {i} ---") # Less verbose
            trainer.model.eval()
            with torch.no_grad(): result = trainer.test() # Assumes trainer.test() exists

            if result and "accuracy" in result:
                acc = result['accuracy']
                print(f"  Test Accuracy (Client {i}): {acc:.2f}%")
                log_data[f"eval_client_{i}_accuracy"] = acc
                all_results[f'client_{i}'] = result
            else:
                print(f"  Client {i} test result format invalid or accuracy missing.")
                all_results[f'client_{i}'] = None

        if wandb.run and log_data: wandb.log(log_data, step=current_round) # Log all client results together
        print("--- End Testing on All Clients ---")
        return all_results


    # ... (debug_print_samples, debug_clients_data, debug_save_samples_images as before) ...
    def debug_print_samples(self, data_manager, subset="train_x", max_per_class=2):
        """Prints sample details for debugging data loading and merging."""
        # ... (implementation as before) ...

    def debug_clients_data(self):
        """Prints class distributions for train, val, test sets of each client."""
        # ... (implementation as before) ...

    def debug_save_samples_images(self, data_manager, subset="train_x", output_dir="debug_samples", max_per_class=5):
        """Copy/save up to `max_per_class` images per class from the given subset."""
        # ... (implementation as before) ...


    # This test method is now less central for federated evaluation, but kept for compatibility
    def test(self, split="test", evaluate_train=False):
        """
        Wrapper for testing. Evaluates client 0's model on its own data.
        Use specific evaluation methods like test_on_unified_dataset for broader eval.
        """
        if not self.clients:
             print("ERROR: No clients available for testing.")
             return None

        print(f"\n--- Running test() method (evaluating Client 0 on its '{split}' split) ---")
        client_model = self.clients[0].model # Use client 0's model

        # Load global weights if they are valid, otherwise use client's current weights
        if self.check_weights_valid(self.global_weights):
            print("  Loading current global weights into Client 0 for testing.")
            try:
                client_model.load_state_dict(self.global_weights, strict=False)
            except Exception as e:
                 print(f"  Warning: Failed to load global weights into Client 0: {e}")
        else:
            print("  Warning: Global weights are invalid. Testing Client 0 with its potentially stale weights.")

        # Call Client 0's internal test method (if it exists and matches this signature)
                        # This relies heavily on the DualPrompt class implementation.
        if hasattr(self.clients[0], 'test'):
            try:
                # The base DualPrompt.test likely doesn't expect 'evaluate_train' directly
                # It usually evaluates on its own dm.test_loader
                # We might need to adapt this if we want Client 0 to evaluate its train set here
                results = self.clients[0].test() # Call the underlying test method
                print(f"--- Finished test() method ---")
                return results
            except Exception as e:
                 print(f"ERROR during Client 0's internal test method call: {e}")
                 import traceback; traceback.print_exc()
                 return None
        else:
             print("ERROR: Client 0 does not have a standard 'test' method to call.")
             return None
        
    def test_on_unified_dataset_eval_only(self):
        """
        Convenience method for train.py to evaluate on the unified dataset.
        This method creates the unified test dataloader and calls test_on_unified_dataset.
        """
        print("\n=== Running evaluation on unified test dataset ===")
        # Create the unified test dataloader
        unified_test_loader = self.create_unified_test_dataloader()
        
        if unified_test_loader:
            # Call the test_on_unified_dataset method with the created loader
            results = self.test_on_unified_dataset(unified_test_loader)
            return results
        else:
            print("ERROR: Failed to create unified test dataloader.")
            return None