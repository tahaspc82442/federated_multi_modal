"""Okay, this is a different partitioning strategy. Instead of partitioning within each dataset, you want to:

Combine all training data from the 5 datasets into one large pool.
Combine all validation data and all test data similarly.
Identify all unique global classes present in the combined training data.
Partition these global classes into N disjoint sets (where N is the total desired number of clients, e.g., 10).
Assign each disjoint set of classes to one client.
Distribute the data points from the combined pools to the clients based on which class they belong to.
This results in N clients total, each holding data from potentially all original datasets but only for a subset of the global classes.

Let's modify build_data_loader to implement this logic.

Key Changes:

Configuration: We will now use cfg.FED.NUM_CLIENTS (e.g., set to 10) to define the total number of clients (N). Remove or ignore cfg.FED.NUM_PARTITIONS_PER_DATASET.
build_data_loader:
Load, standardize classes, and remap to global IDs as before.
Combine: Create combined_train_x, combined_val, combined_test lists by concatenating the remapped data from all datasets.
Partition Global Classes: Get unique global class IDs from combined_train_x. Shuffle them and split into N = cfg.FED.NUM_CLIENTS disjoint chunks.
Map Class to Client: Create a global_class_id_to_client_idx mapping.
Distribute Combined Data: Use a helper function (like the previous partition_data_by_class but adapted) or inline logic to distribute combined_train_x, combined_val, combined_test into N sets based on the global_class_id_to_client_idx map.
Create ClientDataManagers: Create N managers using the distributed data.
Unified Test Loader: Create using the combined_test list."""




import os
import os.path as osp
import torch
import numpy as np
import torch.nn.functional as F
import random
from collections import defaultdict, Counter
from PIL import Image
from tqdm import trange, tqdm
import wandb
import hashlib
import copy

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.data import DataManager
from dassl.data.datasets import Datum
from dassl.utils import (
    mkdir_if_missing,
    load_checkpoint,
    save_checkpoint
)
from dassl.optim import build_lr_scheduler

# Local single-site trainer
from trainers.maple import MaPLe, load_clip_to_cpu # Assuming MaPLe exists
# Custom client data manager
from .client_datamanager import ClientDataManager # Assuming this exists

# Helper function to distribute data based on class->client map
def distribute_data_to_clients(combined_data_list, class_to_client_map, num_clients):
    """
    Distributes a combined list of Datum objects to clients based on class mapping.

    Args:
        combined_data_list (list): List of Datum objects from all datasets.
        class_to_client_map (dict): Maps global class ID to client index.
        num_clients (int): The total number of clients (N).

    Returns:
        list: A list containing num_clients lists, where each sublist
              contains Datum objects belonging to that client.
    """
    client_data = [[] for _ in range(num_clients)]
    unmapped_samples = 0
    for datum in combined_data_list:
        client_idx = class_to_client_map.get(datum.label, -1) # datum.label is global ID
        if client_idx != -1:
            client_data[client_idx].append(datum)
        else:
            # This shouldn't happen if the map is built from train classes
            # and applied to train/val/test, unless val/test have totally unseen classes.
            # Handle assigning these 'unmappable' samples if necessary.
            # Option: assign randomly, assign to client 0, or discard. Let's discard.
            unmapped_samples += 1
            # print(f"Warning: Sample for class {datum.label} ({datum.classname}) has no assigned client. Discarding.")

    if unmapped_samples > 0:
        print(f"Warning: Discarded {unmapped_samples} samples belonging to classes not mapped to any client (likely not present in combined training set).")
    return client_data


@TRAINER_REGISTRY.register()
class MaPLeFederated(TrainerX):
    def __init__(self, cfg):
        self.lab2cname = {}
        self._clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.MAPLE.PREC == "fp16":
            self._clip_model = self._clip_model.half()
        self._clip_model = self._clip_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = next(self._clip_model.parameters()).device
        self.cfg = cfg
        # *** Use total number of clients directly from config ***
        self.num_clients = cfg.FED.NUM_CLIENTS # e.g., 10
        # self.num_partitions_per_dataset = None # Not used anymore
        self.num_rounds = cfg.FED.NUM_ROUNDS
        self.local_epochs = cfg.FED.LOCAL_EPOCHS
        self.clients = []
        self.global_weights = None

        self.nan_stats = {
            "total_updates": 0,
            "failed_clients": [],
            "skipped_rounds": 0
        }
        self.unified_test_loader = None
        self.global_num_classes = 0

        super().__init__(cfg)

    ###################################################
    # A) Build the unified data loader - MODIFIED STRATEGY
    ###################################################
    # def build_data_loader(self):
    #     print("build data loader called - Combined Data Strategy")

    #     # -- 1) Load all datasets (Same as before) --
    #     datasets_to_load = ["PatternNet", "Ucmerced", "EuroSAT", "Mlrs", "Milaid"]
    #     loaded_datasets = {}
    #     loaded_dms = {}
    #     original_lab2cnames = {}
    #     print(f"Loading {len(datasets_to_load)} datasets...")
    #     for name in datasets_to_load:
    #         temp_cfg = self.cfg.clone(); temp_cfg.defrost()
    #         temp_cfg.DATASET.NAME = name; temp_cfg.freeze()
    #         dm = DataManager(temp_cfg)
    #         loaded_datasets[name] = dm.dataset
    #         loaded_dms[name] = dm
    #         original_lab2cnames[name] = dm.lab2cname
    #         print(f"Loaded {name}: train={len(dm.dataset.train_x)}, val={len(dm.dataset.val)}, test={len(dm.dataset.test)}")

    #     # -- 2) Standardize class names (Same as before) --
    #     print("Standardizing class names...")
    #     # (Assuming rename maps are applied as in the previous version)
    #     uc_rename_map = {"tenniscourt": "tennis_court", "golfcourse": "golf_course", "parkinglot": "parking_lot", "storagetanks": "storage_tank", "mobilehomepark": "mobile_home_park", "baseballdiamond": "baseball_field", "denseresidential": "dense_residential", "sparseresidential": "sparse_residential"}
    #     if "Ucmerced" in original_lab2cnames:
    #         uc_lab2cname = original_lab2cnames["Ucmerced"]
    #         for k, old_cname in uc_lab2cname.items(): uc_lab2cname[k] = uc_rename_map.get(old_cname, old_cname)
    #     milaid_rename_map = {"commercial area": "commercial_area", "ice land": "ice_land", "bare land": "bare_land", "detached house": "detached_house", "dry field": "dry_field", "golf course": "golf_course", "ground track field": "ground_track_field", "mobile home park": "mobile_home_park", "oil field": "oil_field", "paddy field": "paddy_field", "parking lot": "parking_lot", "rock land": "rock_land", "solar power plant": "solar_power_plant", "sparse shrub land": "sparse_shrub_land", "storage tank": "storage_tank", "swimming pool": "swimming_pool", "terraced field": "terraced_field", "train station": "train_station", "wastewater plant": "wastewater_plant", "wind turbine": "wind_turbine", "baseball field": "baseball_field", "basketball court": "basketball_court", "tennis court": "tennis_court"}
    #     if "Milaid" in original_lab2cnames:
    #         milaid_lab2cname = original_lab2cnames["Milaid"]
    #         for label, old_cname in milaid_lab2cname.items(): milaid_lab2cname[label] = milaid_rename_map.get(old_cname, old_cname)


    #     # -- 3) Form global list of classes (Same as before) --
    #     print("Creating global class list...")
    #     all_class_sets = [set(cname.lower() for cname in lab2cname.values()) for lab2cname in original_lab2cnames.values()]
    #     global_class_set = set.union(*all_class_sets)
    #     global_list = sorted(list(global_class_set))
    #     self.global_num_classes = len(global_list)
    #     print(f"[INFO] Unified #classes = {self.global_num_classes}")
    #     name2gid = {cname: i for i, cname in enumerate(global_list)}
    #     self.lab2cname = {i: cname for i, cname in enumerate(global_list)}

    #     # -- 4) Remap local labels -> global IDs (Same as before) --
    #     print("Remapping local labels to global IDs...")
    #     def remap(data_list, local_lab2cname):
    #         remapped_list = []
    #         for item in data_list:
    #             cname = local_lab2cname[item.label].lower()
    #             gid = name2gid.get(cname, -1)
    #             if gid != -1: remapped_list.append(Datum(impath=item.impath, label=gid, classname=cname, caption=getattr(item, 'caption', None)))
    #             else: print(f"ERROR: Could not find GID for class '{cname}'. Skipping item: {item.impath}")
    #         return remapped_list

    #     remapped_datasets = {}
    #     for name, dataset in loaded_datasets.items():
    #         # print(f"Remapping {name}...")
    #         remapped_datasets[name] = {"train_x": remap(dataset.train_x, original_lab2cnames[name]), "val": remap(dataset.val, original_lab2cnames[name]), "test": remap(dataset.test, original_lab2cnames[name])}

    #     # -- 5) *** Combine ALL remapped data into single pools *** --
    #     print("Combining data from all datasets...")
    #     combined_train_x = []
    #     combined_val = []
    #     combined_test = []
    #     for name in datasets_to_load: # Use defined order
    #          if name in remapped_datasets:
    #               combined_train_x.extend(remapped_datasets[name]['train_x'])
    #               combined_val.extend(remapped_datasets[name]['val'])
    #               combined_test.extend(remapped_datasets[name]['test'])
    #     print(f"Combined data pools: train={len(combined_train_x)}, val={len(combined_val)}, test={len(combined_test)}")

    #     # Shuffle the combined training data (optional but good practice)
    #     random.shuffle(combined_train_x)

    #     # -- 6) Partition GLOBAL Classes among N clients --
    #     N = self.num_clients # Total number of clients (e.g., 10)
    #     print(f"Partitioning GLOBAL classes among {N} clients...")

    #     # Identify unique GLOBAL class IDs present in the COMBINED training split
    #     unique_global_classes = sorted(list(set(d.label for d in combined_train_x)))
    #     if not unique_global_classes:
    #          raise ValueError("No training samples found in the combined dataset!")

    #     num_unique_global_classes = len(unique_global_classes)
    #     print(f"Found {num_unique_global_classes} unique classes in combined training data.")
    #     if num_unique_global_classes < N:
    #          print(f"Warning: Number of unique classes ({num_unique_global_classes}) is less than the number of clients ({N}). Some clients may get no classes/data.")

    #     # Shuffle and partition the GLOBAL class IDs
    #     random.shuffle(unique_global_classes)
    #     partitioned_global_class_ids = np.array_split(unique_global_classes, N)

    #     # Create a map from global class ID -> client index
    #     global_class_id_to_client_idx = {}
    #     print("Assigning classes to clients:")
    #     for client_idx, class_chunk in enumerate(partitioned_global_class_ids):
    #         print(f"  Client {client_idx}: {len(class_chunk)} classes -> IDs {class_chunk[:5]}...") # Print first 5 classes assigned
    #         for class_id in class_chunk:
    #             global_class_id_to_client_idx[class_id] = client_idx
    #     print(f"Created global class -> client map covering {len(global_class_id_to_client_idx)} classes.")


    #     # -- 7) Distribute Combined Data to Clients based on Class Map --
    #     print(f"Distributing combined data to {N} clients...")
    #     client_train_x = distribute_data_to_clients(combined_train_x, global_class_id_to_client_idx, N)
    #     client_val     = distribute_data_to_clients(combined_val, global_class_id_to_client_idx, N)
    #     client_test    = distribute_data_to_clients(combined_test, global_class_id_to_client_idx, N)


    #     # -- 8) Create N ClientDataManagers --
    #     print(f"Creating {N} ClientDataManagers...")
    #     self.client_data_managers = []
    #     actual_clients_created = 0
    #     for i in range(N):
    #         # Optionally skip clients with no training data
    #         if not client_train_x[i]:
    #             print(f"Warning: Client {i} has no training data after distribution. Skipping client creation.")
    #             continue

    #         print(f"  Creating ClientDataManager for Client {i}: "
    #               f"train={len(client_train_x[i])}, val={len(client_val[i])}, test={len(client_test[i])}")
    #         print(f"    Classes assigned (sample): {[self.lab2cname.get(d.label, d.label) for d in client_train_x[i][:3]]}...")

    #         dm_client = ClientDataManager(
    #             train_x=client_train_x[i],
    #             val=client_val[i],
    #             test=client_test[i],
    #             cfg=self.cfg
    #         )
    #         self.client_data_managers.append(dm_client)
    #         actual_clients_created += 1

    #     # Update the actual number of clients (if some were skipped)
    #     # self.num_clients = actual_clients_created # Update if skipping logic is active
    #     print(f"\nActual ClientDataManagers created: {len(self.client_data_managers)}")
    #     if not self.client_data_managers:
    #         raise ValueError("No ClientDataManagers were created. Check class partitioning and data distribution.")
    #     # It's better if self.num_clients reflects the config target N,
    #     # handle empty clients during training if necessary.

    #     # -- 9) Overwrite cfg.MODEL.NUM_CLASSES (Use Global Count) --
    #     self.cfg.defrost()
    #     self.cfg.MODEL.NUM_CLASSES = self.global_num_classes
    #     self.cfg.freeze()


    #     # -- 10) Create the Unified Test Loader (using the combined_test list) --
    #     print("\nCreating unified test dataloader...")
    #     self.unified_test_data = combined_test # Already combined
    #     print(f"Total samples in unified test data: {len(self.unified_test_data)}")

    #     if self.unified_test_data:
    #         # Use ClientDataManager to easily create the loader
    #         unified_dm = ClientDataManager(train_x=[], val=[], test=self.unified_test_data, cfg=self.cfg)
    #         self.unified_test_loader = unified_dm.test_loader
    #         print(f"Unified test loader created with batch size {self.unified_test_loader.batch_size}")
    #     else:
    #         print("Warning: Unified test data is empty!")
    #         self.unified_test_loader = None


    #     # --- Cleanup ---
    #     self.train_loader_x = None; self.val_loader = None; self.test_loader = None; self.dm = None
    #     print("build_data_loader finished.")


    # ========================================================================
    # Other methods (build_model, train, test_on_unified_dataset, safe_average_weights, etc.)
    # remain largely the same as the previous version (Simple FedAvg, No FedProx, Detailed Viz)
    # because they operate on self.client_data_managers and self.unified_test_loader,
    # which are now populated according to the new strategy.
    # ========================================================================



    # *** CORRECTED build_data_loader with Class Merging & Fix ***
    def build_data_loader(self):
        print("build data loader called - Combined Data Strategy with Class Merging")

        # -- 1) Load all datasets --
        datasets_to_load = ["PatternNet", "Ucmerced", "EuroSAT", "Mlrs", "Milaid"]
        loaded_datasets = {}
        loaded_dms = {}
        original_lab2cnames = {} # Will be modified in place
        print(f"Loading {len(datasets_to_load)} datasets...");
        for name in datasets_to_load:
            temp_cfg = self.cfg.clone(); temp_cfg.defrost(); temp_cfg.DATASET.NAME = name; temp_cfg.freeze(); dm = DataManager(temp_cfg)
            loaded_datasets[name] = dm.dataset; loaded_dms[name] = dm; original_lab2cnames[name] = dm.lab2cname # Store original mapping initially

        # -- 2) Initial Standardization (Spaces, Minor Renames) --
        print("Applying initial class name standardizations...")
        # --- UcMerced Renaming ---
        uc_rename_map = {"tenniscourt": "tennis_court", "golfcourse": "golf_course", "parkinglot": "parking_lot", "storagetanks": "storage_tank", "mobilehomepark": "mobile_home_park", "baseballdiamond": "baseball_diamond", "denseresidential": "dense_residential", "sparseresidential": "sparse_residential"} # Keep diamond temporarily
        if "Ucmerced" in original_lab2cnames:
            for k, old_cname in original_lab2cnames["Ucmerced"].items():
                original_lab2cnames["Ucmerced"][k] = uc_rename_map.get(old_cname, old_cname).lower() # Standardize to lower
        # --- Milaid Renaming (Spaces to Underscores primarily) ---
        milaid_rename_map = {"commercial area": "commercial_area", "ice land": "ice_land", "bare land": "bare_land", "detached house": "detached_house", "dry field": "dry_field", "golf course": "golf_course", "ground track field": "ground_track_field", "mobile home park": "mobile_home_park", "oil field": "oil_field", "paddy field": "paddy_field", "parking lot": "parking_lot", "rock land": "rock_land", "solar power plant": "solar_power_plant", "sparse shrub land": "sparse_shrub_land", "storage tank": "storage_tank", "swimming pool": "swimming_pool", "terraced field": "terraced_field", "train station": "train_station", "wastewater plant": "wastewater_plant", "wind turbine": "wind_turbine", "baseball field": "baseball_field", "basketball court": "basketball_court", "tennis court": "tennis_court"}
        if "Milaid" in original_lab2cnames:
            for k, old_cname in original_lab2cnames["Milaid"].items():
                 original_lab2cnames["Milaid"][k] = milaid_rename_map.get(old_cname, old_cname).lower() # Standardize to lower
        # --- Standardize all others to lowercase ---
        for name, lab2cname_dict in original_lab2cnames.items():
             if name not in ["Ucmerced", "Milaid"]: # Avoid re-lowering
                  for k, v in lab2cname_dict.items():
                      lab2cname_dict[k] = v.lower()

        # -- 2.5) *** Apply Class Merges *** --
        print("Applying class merges...")
        # Map source names (lowercase, after initial standardization) to target names
        class_merge_map = {
            "bareland": "bare_land", # Milaid -> Target
            "dense_residential": "dense_residential_area", # Ucmerced -> Target
            "sparse_residential": "sparse_residential_area", # Ucmerced -> Target
            "harbor": "harbor_port", # Assuming 'harbor' exists -> Target
            "runway_marking": "runway", # Source -> Target (runway should exist)
            "train_station": "railway_station", # Milaid -> Target
            "transformer_station": "substation", # Source -> Target (substation should exist)
            "wastewater_plant": "wastewater_treatment_plant", # Milaid -> Target
            "terrace": "terraced_field", # Source -> Target (terraced_field should exist)
            "parking_space": "parking_lot", # Source -> Target (parking_lot should exist)
            "vegetable_greenhouse": "greenhouse", # Source -> Target (greenhouse should exist)
            "oil_field": "oil_gas_field", # Milaid -> Target
            "oil_well": "oil_gas_field", # Source -> Target
            "quarry": "extractive_site", # Source -> New Target
            "mine": "extractive_site", # Source -> New Target
            "baseball_diamond": "baseball_field", # Ucmerced -> Target (baseball_field exists)
            "solar_panel": "solar_power_plant", # Source -> Target (solar_power_plant exists)
            "overpass": "elevated_crossing", # Source -> New Target
            "viaduct": "elevated_crossing", # Source -> New Target
            "bridge": "elevated_crossing", # Source -> New Target
            "industrial_buildings": "industrial_area", # Source -> Target (industrial_area exists)
            "freeway": "road", # Source -> Target (assuming 'road' class exists post-standardization)
            "parkway": "road", # Source -> Target (assuming 'road' class exists post-standardization)
            "works": "buildings", # Source -> Target (assuming 'building' class exists post-standardization)
        }

        merged_counts = Counter()
        # Apply the merge map to all datasets' lab2cname dictionaries
        for dataset_name, lab2cname_dict in original_lab2cnames.items():
            keys_to_update = list(lab2cname_dict.keys()) # Iterate over copy of keys
            for label in keys_to_update:
                current_name = lab2cname_dict[label] # Already standardized/lowercase
                if current_name in class_merge_map:
                    target_name = class_merge_map[current_name]
                    lab2cname_dict[label] = target_name # Update the name for this label
                    merged_counts[f"{current_name} -> {target_name}"] += 1

        print("Merge Summary:")
        if not merged_counts:
             print("  No classes were merged based on the map.")
        for merge_pair, count in merged_counts.items():
             print(f"  Merged '{merge_pair}' (affected {count} dataset label entries)")


        # -- 3) Form Global List (based on potentially merged names) --
        print("Creating final global class list after merging...")
        all_merged_class_sets = [set(cname for cname in lab2cname.values()) # Already lowercase and merged
                                 for lab2cname in original_lab2cnames.values()]
        global_class_set = set.union(*all_merged_class_sets)
        global_list = sorted(list(global_class_set))
        self.global_num_classes = len(global_list)
        print(f"[INFO] Final Unified #classes after merging = {self.global_num_classes}")
        # Create final mappings
        name2gid = {cname: i for i, cname in enumerate(global_list)}
        self.lab2cname = {i: cname for i, cname in enumerate(global_list)} # GID -> Merged Name


        # -- 4) Remap local labels -> final global IDs --
        print("Remapping data points to final global IDs...")

        def remap_final(data_list, local_lab2cname_merged):
            remapped_list = []
            errors = 0
            for item in data_list:
                original_local_label = item.label # Label from the original dataset load
                merged_cname = local_lab2cname_merged.get(original_local_label)
                if merged_cname:
                    gid = name2gid.get(merged_cname, -1) # Find GID of the merged name
                    if gid != -1: remapped_list.append(Datum(impath=item.impath, label=gid, classname=merged_cname, caption=getattr(item, 'caption', None)))
                    else: errors += 1
                else: errors += 1
            # if errors > 0: print(f"  Remapping Errors: {errors}")
            return remapped_list

        remapped_datasets = {}
        for name, dataset in loaded_datasets.items():
            local_merged_map = original_lab2cnames[name]
            remapped_datasets[name] = {
                "train_x": remap_final(dataset.train_x, local_merged_map),
                "val": remap_final(dataset.val, local_merged_map),
                "test": remap_final(dataset.test, local_merged_map)
                }

        # -- 5) Combine ALL remapped data --
        print("Combining data from all datasets..."); combined_train_x = []; combined_val = []; combined_test = []
        [combined_train_x.extend(remapped_datasets[name]['train_x']) for name in datasets_to_load if name in remapped_datasets]; [combined_val.extend(remapped_datasets[name]['val']) for name in datasets_to_load if name in remapped_datasets]; [combined_test.extend(remapped_datasets[name]['test']) for name in datasets_to_load if name in remapped_datasets]
        print(f"Combined data pools: train={len(combined_train_x)}, val={len(combined_val)}, test={len(combined_test)}"); random.shuffle(combined_train_x)

        # -- 6) Partition GLOBAL Classes among N clients --
        N = self.num_clients
        print(f"Partitioning {self.global_num_classes} GLOBAL classes among {N} clients...");
        unique_global_classes = sorted(list(set(d.label for d in combined_train_x)))
        if not unique_global_classes: raise ValueError("No training samples found in the combined dataset!")
        # *** FIX: Calculate num_unique_global_classes BEFORE using it ***
        num_unique_global_classes = len(unique_global_classes)
        print(f"Found {num_unique_global_classes} unique classes in combined train.");
        # Now check using the calculated value
        if num_unique_global_classes < N: print(f"Warning: {num_unique_global_classes} classes < {N} clients.")

        random.shuffle(unique_global_classes); partitioned_global_class_ids = np.array_split(unique_global_classes, N); global_class_id_to_client_idx = {cid: pidx for pidx, chunk in enumerate(partitioned_global_class_ids) for cid in chunk}


        # -- 7) Distribute Combined Data to Clients --
        print(f"Distributing combined data to {N} clients..."); client_train_x = distribute_data_to_clients(combined_train_x, global_class_id_to_client_idx, N); client_val = distribute_data_to_clients(combined_val, global_class_id_to_client_idx, N); client_test = distribute_data_to_clients(combined_test, global_class_id_to_client_idx, N)

        # -- 8) Create N ClientDataManagers --
        print(f"Creating {N} ClientDataManagers..."); self.client_data_managers = []; actual_clients_created = 0
        for i in range(N):
            if not client_train_x[i]: print(f"Warning: Client {i} has no training data. Skipping."); continue
            dm_client = ClientDataManager(train_x=client_train_x[i], val=client_val[i], test=client_test[i], cfg=self.cfg); self.client_data_managers.append(dm_client); actual_clients_created += 1
        print(f"Actual ClientDataManagers created: {len(self.client_data_managers)}");
        if not self.client_data_managers: raise ValueError("No ClientDataManagers created.")

        # -- 9) Overwrite cfg.MODEL.NUM_CLASSES --
        self.cfg.defrost(); self.cfg.MODEL.NUM_CLASSES = self.global_num_classes; self.cfg.freeze()

        # -- 10) Create the Unified Test Loader --
        print("\nCreating unified test dataloader..."); self.unified_test_data = combined_test; print(f"Total samples in unified test data: {len(self.unified_test_data)}")
        if self.unified_test_data: unified_dm = ClientDataManager(train_x=[], val=[], test=self.unified_test_data, cfg=self.cfg); self.unified_test_loader = unified_dm.test_loader; print(f"Unified test loader created.")
        else: print("Warning: Unified test data empty!"); self.unified_test_loader = None

        # --- Cleanup ---
        self.train_loader_x = None; self.val_loader = None; self.test_loader = None; self.dm = None
        print("build_data_loader finished.")



    ###################################################
    # B) Build local trainers (MaPLe) - (Unchanged conceptually)
    ###################################################
    def build_model(self):
        # This function works as is, iterating over the N ClientDataManagers created above.
        print(f"Building {self.num_clients} client models...") # N clients
        self.clients = []
        global_classnames = list(self.lab2cname.values())
        if not self.client_data_managers: raise RuntimeError("Client DMs not initialized.")

        for i, dm in enumerate(self.client_data_managers):
             # Create trainer for client i (using the ClientDataManager with its subset of classes/data)
             local_trainer = MaPLe(self.cfg, client_id=i, classnames=global_classnames, _clip_model=self._clip_model)
             local_trainer.dm = dm
             local_trainer.build_model()
             self.clients.append(local_trainer)

        if not self.clients: raise RuntimeError("No client trainers created.")
        # Ensure self.clients list matches self.num_clients expected from config
        if len(self.clients) != self.num_clients:
            print(f"Warning: Expected {self.num_clients} clients based on config, but created {len(self.clients)} trainers.")
            # This might happen if some clients had no data and were skipped during DM creation.
            # Adjust self.num_clients or handle potentially fewer clients in training loop.
            # For simplicity, let's assume training loop iterates over len(self.clients).

        self.global_weights = copy.deepcopy(self.clients[0].model.state_dict())
        print(f"Initialized global weights from client 0. Num clients: {len(self.clients)}")


    ###################################################
    # C) Federated training loop - (Unchanged conceptually)
    ###################################################
    def train(self):
            if not self.clients: print("No clients. Exiting."); return
            if not self.unified_test_loader: print("Warning: Unified test loader unavailable.")

            print(f"\nStarting Federated Training with {len(self.clients)} clients...")
            previous_global_weights = copy.deepcopy(self.global_weights) if self.global_weights else None

            for round_idx in trange(self.num_rounds, desc="Federated Rounds"):
                print(f"\n--- Federated Round {round_idx+1}/{self.num_rounds} ---")

                # 1) Broadcast global weights (handle invalid state)
                # (Same robust handling as previous version)
                if self.check_weights_valid(self.global_weights):
                    previous_global_weights = copy.deepcopy(self.global_weights)
                    self.broadcast_weights(self.global_weights)
                else:
                    print(f"!!! Invalid global weights before round {round_idx+1}! Skipping round. !!!")
                    self.nan_stats["skipped_rounds"] += 1
                    if previous_global_weights:
                        print("Attempting to revert..."); self.global_weights = copy.deepcopy(previous_global_weights)
                        if not self.check_weights_valid(self.global_weights): print("!!! Reverting failed! Halting. !!!"); break
                        else: print("Reverted successfully. Skipping this round's training."); continue
                    else: print("!!! Cannot revert! Halting. !!!"); break

                local_state_dicts = []
                valid_clients_indices = []
                round_losses = []

                # 2) Local Training (Simple run_epoch call)
                # (Same as previous version)
                client_pbar = trange(len(self.clients), desc=f"Round {round_idx+1} Clients", leave=False)
                for i in client_pbar:
                    trainer = self.clients[i]; client_pbar.set_description(f"Round {round_idx+1} Client {i}")
                    trainer.epoch = round_idx * self.local_epochs; trainer.max_epoch = (round_idx + 1) * self.local_epochs; last_epoch_loss = 0.0
                    try:
                        for ep in range(trainer.epoch, trainer.max_epoch): epoch_res = trainer.run_epoch(ep); last_epoch_loss = epoch_res.get("loss", 0.0)
                        w = trainer.model.state_dict()
                        if self.check_weights_valid(w): local_state_dicts.append(copy.deepcopy(w)); valid_clients_indices.append(i); round_losses.append(last_epoch_loss)
                        else: print(f"!!! Client {i} invalid weights! Discarding. !!!"); self.nan_stats["failed_clients"].append(i)
                    except Exception as e: print(f"!!! Client {i} failed training: {type(e).__name__} - {str(e)} !!!"); self.nan_stats["failed_clients"].append(i); continue

                # 3) Log average loss
                # (Same as previous version)
                if round_losses: avg_loss_this_round = sum(round_losses) / len(round_losses); print(f"[Round {round_idx+1}] Avg local loss ({len(valid_clients_indices)} clients) = {avg_loss_this_round:.4f}"); wandb.log({"round": round_idx, "avg_loss_across_successful_clients": avg_loss_this_round})
                else: print(f"[Round {round_idx+1}] No clients completed successfully.")

                # 4) Perform FedAvg
                # (Same as previous version)
                if local_state_dicts:
                    print(f"Aggregating weights from {len(local_state_dicts)} clients using Simple FedAvg.")
                    self.global_weights = self.safe_average_weights(local_state_dicts); self.nan_stats['total_updates'] += 1
                    if not self.check_weights_valid(self.global_weights):
                        print(f"!!! Aggregated weights invalid! Reverting. !!!"); self.nan_stats['skipped_rounds'] += 1
                        if previous_global_weights: self.global_weights = copy.deepcopy(previous_global_weights);
                        else: print("!!! Cannot revert! Halting. !!!"); break
                        if not self.check_weights_valid(self.global_weights): print("!!! Reverting failed! Halting. !!!"); break
                        else: print("Reverted successfully.")
                else: print(f"!!! No valid weights to aggregate. Global model unchanged. !!!"); self.nan_stats['skipped_rounds'] += 1

                # 5) Evaluate global model & *** Visualize with Confusion Details ***
                eval_frequency = self.cfg.TEST.EVAL_FREQ if hasattr(self.cfg.TEST, 'EVAL_FREQ') else 1
                save_frequency = self.cfg.SAVE_FREQ if hasattr(self.cfg, 'SAVE_FREQ') else 5
                unified_test_res = {} # Initialize

                if (round_idx + 1) % eval_frequency == 0:
                    if self.unified_test_loader and self.check_weights_valid(self.global_weights):
                        print(f"\n--- Evaluating on Unified Test Set (Round {round_idx+1}) ---")
                        temp_model = self.clients[0].model; temp_model.load_state_dict(self.global_weights); temp_model.eval()

                        # Get detailed results including confusion details
                        unified_test_res = self.test_on_unified_dataset(self.unified_test_loader, temp_model)
                        acc = unified_test_res.get('accuracy', 0.0)
                        class_stats = unified_test_res.get('class_stats', {})
                        confusion_details = unified_test_res.get('confusion_details', {}) # Get the new details

                        # --- Visualization ---
                        print(f"=== Round {round_idx+1} Unified Test Results ===")
                        print(f"Overall Accuracy: {acc:.4f}%")
                        random_guess_acc = (1.0 / self.global_num_classes * 100) if self.global_num_classes > 0 else 0.0
                        print(f"Random Guess Acc: {random_guess_acc:.4f}% ({self.global_num_classes} classes)")

                        print("\n--- Per-Class Performance (with Misclassification Details) ---")
                        if class_stats and confusion_details:
                            # Sort by class ID for consistent order
                            for true_class_id in sorted(class_stats.keys()):
                                stats = class_stats[true_class_id]
                                correct = stats['correct']
                                total = stats['total']
                                class_acc = (correct / total) * 100 if total > 0 else 0.0
                                true_class_name = self.lab2cname.get(true_class_id, f"Unknown GID:{true_class_id}")

                                # Print primary line
                                print(f"  Class {true_class_id:03d} ({true_class_name:<25}): {correct:>4} / {total:<4} ({class_acc:>6.2f}%) Correct")

                                # *** Print Misclassification Details ***
                                if true_class_id in confusion_details and correct < total:
                                    mispredictions = confusion_details[true_class_id]
                                    # Sort mispredictions by count (descending) for clarity
                                    sorted_mispreds = sorted(mispredictions.items(), key=lambda item: item[1], reverse=True)

                                    for pred_class_id, count in sorted_mispreds:
                                        if pred_class_id != true_class_id and count > 0: # Only show actual errors
                                            pred_class_name = self.lab2cname.get(pred_class_id, f"Unknown GID:{pred_class_id}")
                                            print(f"      -> Predicted as '{pred_class_name}' ({pred_class_id:03d}): {count} times")
                                            # Limit number shown? e.g., add break after top 3-5?

                        else: print("  No per-class statistics available.")
                        print("-------------------------------------------------------------")
                        # --- End Visualization ---

                        # Log main accuracy to WandB
                        wandb.log({"round": round_idx, "test_accuracy_unified": acc, "random_guess_accuracy": random_guess_acc})
                        # Optional: Log more detailed stats if needed

                    elif not self.unified_test_loader: print("Skipping evaluation: Loader unavailable.")
                    else: print("Skipping evaluation: Global weights invalid.")

                # Save checkpoint
                if (round_idx + 1) % save_frequency == 0:
                    if self.check_weights_valid(self.global_weights):
                        print(f"\n--- Saving Global Model Checkpoint (Round {round_idx+1}) ---")
                        # Pass results which now might include confusion details if needed in metadata
                        self.save_model(epoch=round_idx + 1, directory=self.cfg.OUTPUT_DIR, val_result=unified_test_res)
                    else: print("Skipping model save: Weights invalid.")

            # 6) Done training
            self.finalize_training()

    # Methods: test_on_unified_dataset, safe_average_weights, check_weights_valid,
    # compute_file_hash, compute_state_dict_hash, broadcast_weights, finalize_training,
    # save_model, load_model, test
    # should be kept from the *previous* response (Simple FedAvg, No FedProx, Viz)
    # as they are compatible with this new data partitioning strategy.

    # Including them again for completeness:

    def test_on_unified_dataset(self, test_loader, model):
            """
            Test the provided model on the unified test dataset loader.
            Computes overall accuracy, per-class correct/total counts,
            and detailed confusion counts.

            Args:
                test_loader: DataLoader for the unified test set.
                model: The model instance with global weights loaded and set to eval mode.

            Returns:
                Dictionary with 'accuracy', 'class_stats', and 'confusion_details'.
                class_stats: {class_id: {'correct': count, 'total': count}}
                confusion_details: {true_label_id: {predicted_label_id: count}}
            """
            device = next(model.parameters()).device
            model.eval()

            total_correct = 0
            total_samples = 0
            # Initialize structures
            class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
            # *** Store detailed confusion counts ***
            confusion_details = defaultdict(lambda: defaultdict(int))
            num_classes = self.global_num_classes

            with torch.no_grad():
                pbar = tqdm(test_loader, desc="Unified Test", leave=False)
                for batch in pbar:
                    images = batch['img'].to(device)
                    labels = batch['label'].to(device) # Global class IDs

                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)

                    # Update overall accuracy count
                    correct_batch = (predicted == labels).sum().item()
                    total_correct += correct_batch
                    total_samples += labels.size(0)

                    # Update per-class stats AND confusion details
                    for i in range(labels.size(0)):
                        true_label = labels[i].item()
                        pred_label = predicted[i].item()

                        if 0 <= true_label < num_classes: # Ensure true label is valid
                            # Update simple correct/total stats
                            class_stats[true_label]['total'] += 1
                            if pred_label == true_label:
                                class_stats[true_label]['correct'] += 1

                            # *** Update detailed confusion count ***
                            if 0 <= pred_label < num_classes: # Ensure prediction is valid class
                                confusion_details[true_label][pred_label] += 1
                            # else: # Handle potential invalid predictions if needed
                            #     confusion_details[true_label]['invalid_pred'] += 1

                    # Update progress bar
                    if total_samples > 0:
                        current_acc = (total_correct / total_samples) * 100
                        pbar.set_postfix({"acc": f"{current_acc:.2f}%"})

            accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0

            return {
                "accuracy": accuracy,
                "class_stats": dict(class_stats), # Convert back to regular dict
                "confusion_details": {k: dict(v) for k, v in confusion_details.items()}, # Convert inner defaultdicts too
                "total_samples": total_samples
            }

    def safe_average_weights(self, local_dicts):
        if not local_dicts: return self.global_weights
        avg_state = {}; num_valid = len(local_dicts); ref_keys = local_dicts[0].keys(); global_keys = self.global_weights.keys() if self.global_weights else set()
        # print(f"Averaging {len(ref_keys)} layers/buffers...")
        for key in ref_keys:
            if self.global_weights and key not in global_keys: continue # Skip keys not in global model
            tensors = [sd[key].float() for sd in local_dicts if key in sd]
            if not tensors:
                 if self.global_weights and key in self.global_weights: avg_state[key] = self.global_weights[key].clone()
                 continue
            try:
                stacked = torch.stack(tensors)
                if torch.isnan(stacked).any() or torch.isinf(stacked).any(): stacked = torch.nan_to_num(stacked, nan=0.0, posinf=1e4, neginf=-1e4)
                avg_tensor = torch.mean(stacked, dim=0)
                if self.global_weights and key in self.global_weights: avg_state[key] = avg_tensor.to(self.global_weights[key].dtype)
                else: avg_state[key] = avg_tensor # Keep float if no global ref
            except RuntimeError as e: print(f"!!! Error averaging key '{key}': {e}. Keeping previous."); avg_state[key] = self.global_weights[key].clone() if self.global_weights and key in self.global_weights else None
        if self.global_weights: # Ensure all global keys are present
            for key in global_keys:
                 if key not in avg_state: print(f"Warning: Global key '{key}' missing in average. Keeping previous."); avg_state[key] = self.global_weights[key].clone()
        return avg_state

    def check_weights_valid(self, state_dict):
        if state_dict is None: print("Weight check failed: state_dict is None."); return False
        for name, param in state_dict.items():
            if torch.isnan(param).any(): print(f"!!! NaN detected in layer '{name}' !!!"); return False
            if torch.isinf(param).any(): print(f"!!! Inf detected in layer '{name}' !!!"); return False
        return True

    def compute_file_hash(self, path):
        sha256 = hashlib.sha256();
        try:
            with open(path, 'rb') as f:
                while chunk := f.read(8192): sha256.update(chunk)
            return sha256.hexdigest()
        except FileNotFoundError: print(f"Error hashing: File not found at {path}"); return None

    def compute_state_dict_hash(self, state_dict):
        sha256 = hashlib.sha256();
        for key in sorted(state_dict.keys()): sha256.update(state_dict[key].cpu().numpy().tobytes())
        return sha256.hexdigest()

    def broadcast_weights(self, global_sd):
        if not self.check_weights_valid(global_sd): raise ValueError("Invalid weights broadcast.")
        num_clients_broadcasted = 0
        for i, client_trainer in enumerate(self.clients):
            try:
                missing_keys, unexpected_keys = client_trainer.model.load_state_dict(global_sd, strict=False)
                if missing_keys: print(f"Warning Client {i}: Missing keys: {missing_keys}")
                if unexpected_keys: print(f"Warning Client {i}: Unexpected keys: {unexpected_keys}")
                client_trainer.optim.state = defaultdict(dict)
                if hasattr(client_trainer, 'sched') and client_trainer.sched is not None:
                    client_trainer.sched = build_lr_scheduler(client_trainer.optim, client_trainer.cfg.OPTIM)
                    current_round_start_epoch = client_trainer.epoch
                    client_trainer.sched.last_epoch = current_round_start_epoch - 1
                num_clients_broadcasted += 1
            except Exception as e: print(f"!!! Error broadcasting/resetting client {i}: {e} !!!")
        # print(f"Broadcasted weights to {num_clients_broadcasted}/{len(self.clients)} clients.")

    def finalize_training(self):
            print("\n" + "="*40 + "\n          Federated Training Finished\n" + "="*40)
            print("\nTraining Summary:") # Same summary stats as before
            print(f"Total Rounds Configured: {self.num_rounds}"); print(f"Successful Aggregation Rounds: {self.nan_stats['total_updates']}"); print(f"Skipped/Failed Aggregation Rounds: {self.nan_stats['skipped_rounds']}")
            unique_failed_clients = set(idx for idx in self.nan_stats['failed_clients'] if idx < len(self.clients)); print(f"Unique Clients Encountering Failures: {len(unique_failed_clients)} / {len(self.clients)}")

            print("\n--- Final Evaluation on Unified Test Set ---"); final_test_res = {}
            if self.unified_test_loader and self.check_weights_valid(self.global_weights):
                final_model = self.clients[0].model; final_model.load_state_dict(self.global_weights); final_model.eval()
                final_test_res = self.test_on_unified_dataset(self.unified_test_loader, final_model) # Get detailed results
                final_acc = final_test_res.get('accuracy', 0.0); class_stats = final_test_res.get('class_stats', {}); confusion_details = final_test_res.get('confusion_details', {})
                print(f"Final Unified Test Accuracy: {final_acc:.4f}%")
                random_guess_acc = (1.0 / self.global_num_classes * 100) if self.global_num_classes > 0 else 0.0; print(f"Final Random Guess Acc:    {random_guess_acc:.4f}%")
                wandb.log({"final_unified_test_accuracy": final_acc}); wandb.summary["final_unified_test_accuracy"] = final_acc

                print("\n--- Final Per-Class Performance (with Misclassification Details) ---")
                if class_stats and confusion_details:
                    for true_class_id in sorted(class_stats.keys()):
                        stats = class_stats[true_class_id]; correct, total = stats['correct'], stats['total']
                        class_acc = (correct / total * 100) if total > 0 else 0.0
                        true_class_name = self.lab2cname.get(true_class_id, f"Unknown GID:{true_class_id}")
                        print(f"  Class {true_class_id:03d} ({true_class_name:<25}): {correct:>4} / {total:<4} ({class_acc:>6.2f}%) Correct")
                        # Print misclassifications
                        if true_class_id in confusion_details and correct < total:
                            mispredictions = confusion_details[true_class_id]; sorted_mispreds = sorted(mispredictions.items(), key=lambda item: item[1], reverse=True)
                            shown_mispreds = 0
                            for pred_class_id, count in sorted_mispreds:
                                if pred_class_id != true_class_id and count > 0:
                                    pred_class_name = self.lab2cname.get(pred_class_id, f"Unknown GID:{pred_class_id}")
                                    print(f"      -> Predicted as '{pred_class_name}' ({pred_class_id:03d}): {count} times")
                                    shown_mispreds +=1
                                    if shown_mispreds >= 5: print("      -> ... (Top 5 mispredictions shown)"); break # Limit output per class

                else: print("  No per-class statistics available.")
                print("-------------------------------------------------------------")

            else: print("Skipping final evaluation: Loader missing or weights invalid.")
            print("\n--- Saving Final Global Model ---")
            if self.check_weights_valid(self.global_weights):
                self.save_model(epoch=self.num_rounds, directory=self.cfg.OUTPUT_DIR, is_best=False, val_result=final_test_res) # Pass full results
                print(f"Final model saved to {self.cfg.OUTPUT_DIR}"); wandb.summary["final_model_saved"] = True
            else: print("Skipping final model save: Weights invalid."); wandb.summary["final_model_saved"] = False


    def save_model(self, epoch=None, directory="", is_best=False, val_result=None):
        if not directory:
            directory = self.cfg.OUTPUT_DIR
        mkdir_if_missing(directory)

        subfolder = "MultiModalPromptLearner_Aggregator"
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


    def load_model(self, directory, epoch=None, expected_file_hash=None):
        """Load aggregator weights with sanity checks."""
        if not directory:
            print("Skipping load_model, no pretrained path given")
            return

        subfolder = "MultiModalPromptLearner_Aggregator"
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

    def test(self, evaluate_train=False):
        print("\n--- Running Final Test using Unified Test Loader ---")
        if self.unified_test_loader and self.check_weights_valid(self.global_weights):
            model_to_test = self.clients[0].model; model_to_test.load_state_dict(self.global_weights); model_to_test.eval()
            results = self.test_on_unified_dataset(self.unified_test_loader, model_to_test)
            print(f"Final Unified Test Results: Accuracy = {results.get('accuracy', 0.0):.4f}%")
            return results
        else: print("Cannot run final test: Loader missing or weights invalid."); return {"accuracy": 0.0, "class_stats": {}}