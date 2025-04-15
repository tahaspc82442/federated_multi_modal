
"""Maple fed with 50 clients
"""






import os
import os.path as osp
import torch
import numpy as np
import torch.nn.functional as F
import random # Added for shuffling classes
from collections import defaultdict, Counter
from PIL import Image
from tqdm import trange, tqdm # Added tqdm for test loader
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
# from .debug import debug_collate # Assuming this exists if needed

# Helper function (unchanged)
def partition_data_by_class(data_list, class_to_partition_map, num_partitions):
    partitioned_data = [[] for _ in range(num_partitions)]
    unknown_class_samples = 0
    for datum in data_list:
        partition_idx = class_to_partition_map.get(datum.label, -1)
        if partition_idx != -1:
            partitioned_data[partition_idx].append(datum)
        else:
            partitioned_data[0].append(datum)
            unknown_class_samples += 1
    # if unknown_class_samples > 0:
    #     print(f"Warning: Assigned {unknown_class_samples} samples with unseen classes during partitioning to partition 0.")
    return partitioned_data


@TRAINER_REGISTRY.register()
class MaPLeFederated(TrainerX):
    def __init__(self, cfg):
        self.lab2cname = {}
        self._clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.MAPLE.PREC == "fp16":
            self._clip_model = self._clip_model.half()
        self._clip_model = self._clip_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.device = next(self._clip_model.parameters()).device # Store device
        self.cfg = cfg
        self.num_partitions_per_dataset = cfg.FED.NUM_PARTITIONS_PER_DATASET
        self.num_datasets = 5 # Hardcoded for now
        self.num_clients = self.num_datasets * self.num_partitions_per_dataset
        self.num_rounds = cfg.FED.NUM_ROUNDS
        self.local_epochs = cfg.FED.LOCAL_EPOCHS
        self.clients = []
        self.global_weights = None
        # self.prox_mu = 0.0 # Removed FedProx

        self.nan_stats = {
            "total_updates": 0,
            "failed_clients": [],
            "skipped_rounds": 0
        }
        self.unified_test_loader = None
        self.global_num_classes = 0 # Will be set in build_data_loader

        super().__init__(cfg) # Calls self.build_data_loader

    ###################################################
    # A) Build the unified data loader
    ###################################################
    def build_data_loader(self):
        print("build data loader called")

        # -- 1) Load all datasets (Same as before) --
        datasets_to_load = ["PatternNet", "Ucmerced", "EuroSAT", "Mlrs", "Milaid"]
        loaded_datasets = {}
        loaded_dms = {}
        original_lab2cnames = {}
        print(f"Loading {len(datasets_to_load)} datasets...")
        for name in datasets_to_load:
            temp_cfg = self.cfg.clone(); temp_cfg.defrost()
            temp_cfg.DATASET.NAME = name; temp_cfg.freeze()
            dm = DataManager(temp_cfg)
            loaded_datasets[name] = dm.dataset
            loaded_dms[name] = dm
            original_lab2cnames[name] = dm.lab2cname
            print(f"Loaded {name}: train={len(dm.dataset.train_x)}, val={len(dm.dataset.val)}, test={len(dm.dataset.test)}")

        # -- 2) Standardize class names (Same as before) --
        print("Standardizing class names...")
        # --- UcMerced Renaming ---
        uc_rename_map = {"tenniscourt": "tennis_court", "golfcourse": "golf_course", "parkinglot": "parking_lot", "storagetanks": "storage_tank", "mobilehomepark": "mobile_home_park", "baseballdiamond": "baseball_field", "denseresidential": "dense_residential", "sparseresidential": "sparse_residential"}
        if "Ucmerced" in original_lab2cnames:
            uc_lab2cname = original_lab2cnames["Ucmerced"]
            for k, old_cname in uc_lab2cname.items(): uc_lab2cname[k] = uc_rename_map.get(old_cname, old_cname)
        # --- Milaid Renaming ---
        milaid_rename_map = {"commercial area": "commercial_area", "ice land": "ice_land", "bare land": "bare_land", "detached house": "detached_house", "dry field": "dry_field", "golf course": "golf_course", "ground track field": "ground_track_field", "mobile home park": "mobile_home_park", "oil field": "oil_field", "paddy field": "paddy_field", "parking lot": "parking_lot", "rock land": "rock_land", "solar power plant": "solar_power_plant", "sparse shrub land": "sparse_shrub_land", "storage tank": "storage_tank", "swimming pool": "swimming_pool", "terraced field": "terraced_field", "train station": "train_station", "wastewater plant": "wastewater_plant", "wind turbine": "wind_turbine", "baseball field": "baseball_field", "basketball court": "basketball_court", "tennis court": "tennis_court"}
        if "Milaid" in original_lab2cnames:
            milaid_lab2cname = original_lab2cnames["Milaid"]
            for label, old_cname in milaid_lab2cname.items(): milaid_lab2cname[label] = milaid_rename_map.get(old_cname, old_cname)

        # -- 3) Form global list of classes (Same as before) --
        print("Creating global class list...")
        all_class_sets = [set(cname.lower() for cname in lab2cname.values()) for lab2cname in original_lab2cnames.values()]
        global_class_set = set.union(*all_class_sets)
        global_list = sorted(list(global_class_set))
        self.global_num_classes = len(global_list) # Store global num classes
        print(f"[INFO] Unified #classes = {self.global_num_classes}")
        name2gid = {cname: i for i, cname in enumerate(global_list)}
        self.lab2cname = {i: cname for i, cname in enumerate(global_list)}

        # -- 4) Remap local labels -> global IDs (Same as before) --
        print("Remapping local labels to global IDs...")
        def remap(data_list, local_lab2cname):
            remapped_list = []
            for item in data_list:
                cname = local_lab2cname[item.label].lower()
                gid = name2gid.get(cname, -1)
                if gid != -1: remapped_list.append(Datum(impath=item.impath, label=gid, classname=cname, caption=getattr(item, 'caption', None)))
                else: print(f"ERROR: Could not find GID for class '{cname}'. Skipping item: {item.impath}")
            return remapped_list

        remapped_datasets = {}
        for name, dataset in loaded_datasets.items():
            print(f"Remapping {name}...")
            remapped_datasets[name] = {"train_x": remap(dataset.train_x, original_lab2cnames[name]), "val": remap(dataset.val, original_lab2cnames[name]), "test": remap(dataset.test, original_lab2cnames[name])}
            print(f"Remapped {name}: train={len(remapped_datasets[name]['train_x'])}, val={len(remapped_datasets[name]['val'])}, test={len(remapped_datasets[name]['test'])}")

        # -- 5) Overwrite cfg.MODEL.NUM_CLASSES (Same as before) --
        self.cfg.defrost()
        self.cfg.MODEL.NUM_CLASSES = self.global_num_classes
        self.cfg.freeze()

        # -- 6) Partition each dataset and create ClientDataManagers (Same as before) --
        print(f"Partitioning each dataset into {self.num_partitions_per_dataset} clients...")
        self.client_data_managers = []
        client_global_idx = 0
        for name, data_splits in remapped_datasets.items():
            print(f"-- Partitioning {name} --")
            unique_classes_in_dataset = sorted(list(set(d.label for d in data_splits['train_x'])))
            if not unique_classes_in_dataset: print(f"Warning: Dataset {name} has no training samples. Skipping."); continue
            print(f"{name}: Found {len(unique_classes_in_dataset)} unique classes in training data.")
            random.shuffle(unique_classes_in_dataset)
            partitioned_class_ids = np.array_split(unique_classes_in_dataset, self.num_partitions_per_dataset)
            class_id_to_partition_map = {cid: pidx for pidx, chunk in enumerate(partitioned_class_ids) for cid in chunk}
            print(f"  Partition map created for {len(class_id_to_partition_map)} classes.")

            partitioned_train_data = partition_data_by_class(data_splits['train_x'], class_id_to_partition_map, self.num_partitions_per_dataset)
            partitioned_val_data = partition_data_by_class(data_splits['val'], class_id_to_partition_map, self.num_partitions_per_dataset)
            partitioned_test_data = partition_data_by_class(data_splits['test'], class_id_to_partition_map, self.num_partitions_per_dataset)

            for i in range(self.num_partitions_per_dataset):
                if not partitioned_train_data[i]: print(f"Warning: Client {client_global_idx} (Dataset {name}, Partition {i}) has no training data. Skipping."); continue
                print(f"  Creating ClientDataManager {client_global_idx} (DS: {name}, P: {i}): tr={len(partitioned_train_data[i])}, v={len(partitioned_val_data[i])}, te={len(partitioned_test_data[i])}")
                dm_client = ClientDataManager(train_x=partitioned_train_data[i], val=partitioned_val_data[i], test=partitioned_test_data[i], cfg=self.cfg)
                self.client_data_managers.append(dm_client)
                client_global_idx += 1

        self.num_clients = len(self.client_data_managers)
        print(f"\nTotal clients created: {self.num_clients}")
        if self.num_clients == 0: raise ValueError("No clients created.")

        # -- 7) Create the Unified Test Loader (Same as before) --
        print("\nCreating unified test dataloader...")
        self.unified_test_data = [item for name in datasets_to_load if name in remapped_datasets for item in remapped_datasets[name]['test']]
        print(f"Total samples in unified test data: {len(self.unified_test_data)}")
        if self.unified_test_data:
            unified_dm = ClientDataManager(train_x=[], val=[], test=self.unified_test_data, cfg=self.cfg)
            self.unified_test_loader = unified_dm.test_loader
            print(f"Unified test loader created with batch size {self.unified_test_loader.batch_size}")
        else: print("Warning: Unified test data is empty!")

        # -- Cleanup (Same as before) --
        self.train_loader_x = None; self.val_loader = None; self.test_loader = None; self.dm = None
        print("build_data_loader finished.")


    ###################################################
    # B) Build local trainers (MaPLe) - (Unchanged)
    ###################################################
    def build_model(self):
        print(f"Building {self.num_clients} client models...")
        self.clients = []
        global_classnames = list(self.lab2cname.values())
        if not self.client_data_managers: raise RuntimeError("Client DMs not initialized.")

        for i, dm in enumerate(self.client_data_managers):
            local_trainer = MaPLe(self.cfg, client_id=i, classnames=global_classnames, _clip_model=self._clip_model)
            local_trainer.dm = dm
            local_trainer.build_model()
            self.clients.append(local_trainer)

        if not self.clients: raise RuntimeError("No client trainers created.")
        self.global_weights = copy.deepcopy(self.clients[0].model.state_dict())
        print(f"Initialized global weights from client 0. Num clients: {len(self.clients)}")


    ###################################################
    # C) Federated training loop
    ###################################################
    def train(self):
        if not self.clients: print("No clients. Exiting."); return
        if not self.unified_test_loader: print("Warning: Unified test loader unavailable.")

        print(f"\nStarting Federated Training with {len(self.clients)} clients...")
        # Store previous weights for safety check after aggregation
        previous_global_weights = copy.deepcopy(self.global_weights) if self.global_weights else None

        for round_idx in trange(self.num_rounds, desc="Federated Rounds"):
            print(f"\n--- Federated Round {round_idx+1}/{self.num_rounds} ---")

            # 1) Broadcast global weights
            if self.check_weights_valid(self.global_weights):
                previous_global_weights = copy.deepcopy(self.global_weights) # Store good weights before broadcast
                self.broadcast_weights(self.global_weights) # FedProx store removed
            else:
                print(f"!!! Invalid global weights before round {round_idx+1}! Skipping round. !!!")
                self.nan_stats["skipped_rounds"] += 1
                # Try reverting? For now, just skip.
                if previous_global_weights:
                    print("Attempting to revert to previous valid global weights...")
                    self.global_weights = copy.deepcopy(previous_global_weights)
                    if not self.check_weights_valid(self.global_weights):
                        print("!!! Reverting failed! Previous weights also invalid? Halting. !!!")
                        break # Stop training if weights corrupted beyond repair
                    else:
                        print("Reverted successfully. Skipping broadcast and training for this round.")
                        continue # Skip training phase this round
                else:
                    print("!!! Cannot revert, no previous valid weights. Halting. !!!")
                    break # Stop if initial weights are bad or no history

            local_state_dicts = []
            valid_clients_indices = []
            round_losses = []

            # 2) Local Training
            client_pbar = trange(len(self.clients), desc=f"Round {round_idx+1} Clients", leave=False)
            for i in client_pbar:
                trainer = self.clients[i]
                client_pbar.set_description(f"Round {round_idx+1} Client {i}")
                trainer.epoch = round_idx * self.local_epochs
                trainer.max_epoch = (round_idx + 1) * self.local_epochs
                last_epoch_loss = 0.0

                try:
                    for ep in range(trainer.epoch, trainer.max_epoch):
                        # *** MODIFIED CALL: Removed FedProx params ***
                        epoch_res = trainer.run_epoch(ep) # Removed prox args
                        last_epoch_loss = epoch_res.get("loss", 0.0)

                    w = trainer.model.state_dict()
                    if self.check_weights_valid(w):
                        local_state_dicts.append(copy.deepcopy(w))
                        valid_clients_indices.append(i)
                        round_losses.append(last_epoch_loss)
                        # wandb.log(...) # Keep wandb logging if desired
                    else:
                        print(f"!!! Client {i} produced invalid weights! Discarding. !!!")
                        self.nan_stats["failed_clients"].append(i)

                except Exception as e: # Catch broader exceptions
                     print(f"!!! Client {i} failed training with Exception: {type(e).__name__} - {str(e)} !!!")
                     self.nan_stats["failed_clients"].append(i)
                     continue

            # 3) Log average loss
            if round_losses:
                avg_loss_this_round = sum(round_losses) / len(round_losses)
                print(f"[Round {round_idx+1}] Avg local loss (last epoch, {len(valid_clients_indices)} clients) = {avg_loss_this_round:.4f}")
                wandb.log({"round": round_idx, "avg_loss_across_successful_clients": avg_loss_this_round})
            else: print(f"[Round {round_idx+1}] No clients completed successfully.")

            # 4) Perform FedAvg
            if local_state_dicts:
                print(f"Aggregating weights from {len(local_state_dicts)} clients using Simple FedAvg.")
                # Pass only the state dicts for simple averaging
                self.global_weights = self.safe_average_weights(local_state_dicts)
                self.nan_stats['total_updates'] += 1

                # Verify aggregated weights
                if not self.check_weights_valid(self.global_weights):
                     print(f"!!! Aggregated global weights are invalid after round {round_idx+1}! Reverting. !!!")
                     self.nan_stats['skipped_rounds'] += 1
                     if previous_global_weights: # Revert to weights before this round's broadcast
                          self.global_weights = copy.deepcopy(previous_global_weights)
                          if not self.check_weights_valid(self.global_weights):
                               print("!!! Reverting failed! Previous weights also invalid? Halting. !!!")
                               break
                          else: print("Reverted to previous valid weights.")
                     else:
                          print("!!! Cannot revert, no previous valid weights. Halting. !!!")
                          break
                # else: # Store the newly aggregated valid weights as the previous for the *next* round
                     # previous_global_weights = copy.deepcopy(self.global_weights)
                     # This happens naturally at the start of the next round now.

            else:
                print(f"!!! No valid local weights to aggregate in round {round_idx+1}. Global model unchanged. !!!")
                self.nan_stats['skipped_rounds'] += 1
                # Global weights remain 'previous_global_weights'

            # 5) Evaluate global model on UNIFIED test set & Visualize
            eval_frequency = self.cfg.TEST.EVAL_FREQ if hasattr(self.cfg.TEST, 'EVAL_FREQ') else 1
            save_frequency = self.cfg.SAVE_FREQ if hasattr(self.cfg, 'SAVE_FREQ') else 5

            if (round_idx + 1) % eval_frequency == 0:
                 if self.unified_test_loader and self.check_weights_valid(self.global_weights):
                     print(f"\n--- Evaluating on Unified Test Set (Round {round_idx+1}) ---")
                     temp_model = self.clients[0].model # Use structure from client 0
                     temp_model.load_state_dict(self.global_weights)
                     temp_model.eval()

                     # *** Get DETAILED results ***
                     unified_test_res = self.test_on_unified_dataset(self.unified_test_loader, temp_model)
                     acc = unified_test_res.get('accuracy', 0.0)
                     class_stats = unified_test_res.get('class_stats', {})

                     # --- Visualization ---
                     print(f"=== Round {round_idx+1} Unified Test Results ===")
                     print(f"Overall Accuracy: {acc:.4f}%")

                     # Calculate Random Guess Accuracy
                     if self.global_num_classes > 0:
                         random_guess_acc = (1.0 / self.global_num_classes) * 100
                         print(f"Random Guess Acc: {random_guess_acc:.4f}% ({self.global_num_classes} classes)")
                     else:
                         print("Random Guess Acc: N/A (0 classes)")

                     print("\n--- Per-Class Performance ---")
                     correct_total = 0
                     samples_total = 0
                     if class_stats:
                          # Sort by class ID for consistent order
                          for class_id in sorted(class_stats.keys()):
                              stats = class_stats[class_id]
                              correct = stats['correct']
                              total = stats['total']
                              class_acc = (correct / total) * 100 if total > 0 else 0.0
                              class_name = self.lab2cname.get(class_id, f"Unknown GID:{class_id}")
                              print(f"  Class {class_id:03d} ({class_name:<25}): {correct:>4} / {total:<4} ({class_acc:>6.2f}%) Correct")
                              correct_total += correct
                              samples_total += total
                          # Sanity check totals
                          print(f"  Check: Total Correct={correct_total}, Total Samples={samples_total}, Overall Acc={ (correct_total/samples_total*100) if samples_total else 0 :.4f}%")
                     else:
                          print("  No per-class statistics available.")
                     print("-----------------------------")
                     # --- End Visualization ---

                     wandb.log({
                         "round": round_idx,
                         "test_accuracy_unified": acc,
                         "random_guess_accuracy": random_guess_acc if self.global_num_classes > 0 else 0
                         # Optional: Log per-class stats (can be very verbose)
                         # **{f"class_{self.lab2cname.get(cid)}_acc": (stats['correct']/stats['total']*100 if stats['total']>0 else 0)
                         #   for cid, stats in class_stats.items()}
                     })
                 # ... (rest of evaluation conditions)

            # Save checkpoint (unchanged)
            if (round_idx + 1) % save_frequency == 0:
                 if self.check_weights_valid(self.global_weights):
                     print(f"\n--- Saving Global Model Checkpoint (Round {round_idx+1}) ---")
                     self.save_model(epoch=round_idx + 1, directory=self.cfg.OUTPUT_DIR,
                                      val_result=unified_test_res if 'unified_test_res' in locals() else None) # Save test results if available
                 else: print("Skipping model save: Global weights invalid.")

        # 6) Done training
        self.finalize_training()


    def test_on_unified_dataset(self, test_loader, model):
        """
        Test the provided model on the unified test dataset loader.
        Computes overall accuracy and per-class correct/total counts.

        Args:
            test_loader: DataLoader for the unified test set.
            model: The model instance with global weights loaded and set to eval mode.

        Returns:
            Dictionary with 'accuracy' and 'class_stats'
            where class_stats is {class_id: {'correct': count, 'total': count}}.
        """
        device = next(model.parameters()).device
        model.eval()

        total_correct = 0
        total_samples = 0
        # *** Initialize per-class stats dictionary ***
        class_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        num_classes = self.global_num_classes # Get total number of classes

        with torch.no_grad():
            pbar = tqdm(test_loader, desc="Unified Test", leave=False)
            for batch in pbar:
                images = batch['img'].to(device)
                labels = batch['label'].to(device) # These are global class IDs

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                # Update overall accuracy count
                correct_batch = (predicted == labels).sum().item()
                total_correct += correct_batch
                total_samples += labels.size(0)

                # *** Update per-class stats ***
                for i in range(labels.size(0)):
                    label = labels[i].item()
                    pred = predicted[i].item()
                    if 0 <= label < num_classes: # Ensure label is valid
                        class_stats[label]['total'] += 1
                        if pred == label:
                            class_stats[label]['correct'] += 1
                    # else: print(f"Warning: Encountered invalid label {label} during testing.")

                # Update progress bar
                if total_samples > 0:
                     current_acc = (total_correct / total_samples) * 100
                     pbar.set_postfix({"acc": f"{current_acc:.2f}%"})

        accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0

        return {
            "accuracy": accuracy,
            "class_stats": dict(class_stats), # Convert back to regular dict
            "total_samples": total_samples
        }

    # *** Use Simple FedAvg ***
    def safe_average_weights(self, local_dicts):
        """Averages weights using simple FedAvg."""
        if not local_dicts:
            print("Warning: safe_average_weights called with empty list. Returning previous global weights.")
            return self.global_weights

        avg_state = {}
        num_valid = len(local_dicts)
        # Use keys from the first dictionary as reference
        ref_keys = local_dicts[0].keys()
        global_keys = self.global_weights.keys() if self.global_weights else set()

        print(f"Averaging {len(ref_keys)} layers/buffers...")

        for key in ref_keys:
            # Ensure the key exists in the global model for dtype reference (if global exists)
            if self.global_weights and key not in global_keys:
                 print(f"Warning: Key '{key}' found in local dict but not global state. Skipping.")
                 continue

            # Collect tensors for this key, ensuring they are float32 for stable averaging
            tensors = []
            for sd in local_dicts:
                 if key in sd:
                      tensors.append(sd[key].float())
                 else:
                      print(f"Warning: Key '{key}' missing in a local state dict during averaging. Skipping that dict for this key.")

            if not tensors: # Skip if key was missing in all dicts
                 if self.global_weights and key in self.global_weights:
                      avg_state[key] = self.global_weights[key].clone() # Keep previous global value
                 continue

            # Stack and average
            try:
                 stacked = torch.stack(tensors)
                 # Robust check for NaNs/Infs before averaging
                 if torch.isnan(stacked).any() or torch.isinf(stacked).any():
                      print(f"!!! NaN/Inf detected in tensors for key '{key}' BEFORE averaging. Attempting nan_to_num. !!!")
                      stacked = torch.nan_to_num(stacked, nan=0.0, posinf=1e4, neginf=-1e4) # Replace bad values

                 avg_tensor = torch.mean(stacked, dim=0)

                 # Convert back to original dtype if global weights exist and have the key
                 if self.global_weights and key in self.global_weights:
                      original_dtype = self.global_weights[key].dtype
                      avg_state[key] = avg_tensor.to(original_dtype)
                 else: # If no global reference, keep float or try inferring? Default to float.
                      avg_state[key] = avg_tensor
                      print(f"Warning: No global reference for key '{key}'. Keeping aggregated tensor as float.")

            except RuntimeError as e:
                 print(f"!!! Error during averaging for key '{key}': {e}. Keeping previous global value if possible. !!!")
                 if self.global_weights and key in self.global_weights:
                      avg_state[key] = self.global_weights[key].clone() # Fallback


        # Ensure all keys from the global model are present in the new state
        # (Handles cases where a key might not have been present in any local dict)
        if self.global_weights:
             for key in global_keys:
                  if key not in avg_state:
                      print(f"Warning: Key '{key}' from global state was not averaged (missing in local dicts?). Keeping previous value.")
                      avg_state[key] = self.global_weights[key].clone()

        return avg_state


    def _calculate_diversity(self, client):
         # (This function is no longer called by safe_average_weights but kept for reference)
         pass


    def check_weights_valid(self, state_dict):
        # (Unchanged)
        if state_dict is None: print("Weight check failed: state_dict is None."); return False
        for name, param in state_dict.items():
            if torch.isnan(param).any(): print(f"!!! NaN detected in layer '{name}' !!!"); return False
            if torch.isinf(param).any(): print(f"!!! Inf detected in layer '{name}' !!!"); return False
        return True


    def compute_file_hash(self, path):
        # (Unchanged)
        sha256 = hashlib.sha256();
        try:
            with open(path, 'rb') as f:
                while chunk := f.read(8192): sha256.update(chunk)
            return sha256.hexdigest()
        except FileNotFoundError: print(f"Error hashing: File not found at {path}"); return None


    def compute_state_dict_hash(self, state_dict):
        # (Unchanged)
        sha256 = hashlib.sha256();
        for key in sorted(state_dict.keys()): sha256.update(state_dict[key].cpu().numpy().tobytes())
        return sha256.hexdigest()


    # *** REMOVED FedProx logic ***
    def broadcast_weights(self, global_sd):
        """Broadcast global state dict to all clients."""
        if not self.check_weights_valid(global_sd):
             raise ValueError("Attempting to broadcast invalid global weights.")

        num_clients_broadcasted = 0
        for i, client_trainer in enumerate(self.clients):
            try:
                missing_keys, unexpected_keys = client_trainer.model.load_state_dict(global_sd, strict=False)
                if missing_keys: print(f"Warning Client {i}: Missing keys during broadcast: {missing_keys}")
                if unexpected_keys: print(f"Warning Client {i}: Unexpected keys: {unexpected_keys}")

                # --- REMOVED storing global_model_params_start_round ---

                # Reset optimizer state
                client_trainer.optim.state = defaultdict(dict)

                # Rebuild or reset scheduler
                if hasattr(client_trainer, 'sched') and client_trainer.sched is not None:
                    client_trainer.sched = build_lr_scheduler(client_trainer.optim, client_trainer.cfg.OPTIM)
                    current_round_start_epoch = client_trainer.epoch # epoch is set before run_epoch call
                    client_trainer.sched.last_epoch = current_round_start_epoch - 1
                num_clients_broadcasted += 1
            except Exception as e: print(f"!!! Error broadcasting/resetting client {i}: {e} !!!")
        # print(f"Broadcasted weights to {num_clients_broadcasted}/{len(self.clients)} clients.")


    def finalize_training(self):
        # (Enhanced final logging)
        print("\n" + "="*40 + "\n          Federated Training Finished\n" + "="*40)
        print("\nTraining Summary:")
        print(f"Total Rounds Configured: {self.num_rounds}")
        print(f"Successful Aggregation Rounds: {self.nan_stats['total_updates']}")
        print(f"Skipped/Failed Aggregation Rounds: {self.nan_stats['skipped_rounds']}")
        unique_failed_clients = set(idx for idx in self.nan_stats['failed_clients'] if idx < self.num_clients) # Filter out potential old indices if num_clients changed
        print(f"Unique Clients Encountering Failures: {len(unique_failed_clients)} / {self.num_clients}")

        print("\n--- Final Evaluation on Unified Test Set ---")
        if self.unified_test_loader and self.check_weights_valid(self.global_weights):
            final_model = self.clients[0].model
            final_model.load_state_dict(self.global_weights)
            final_model.eval()
            final_test_res = self.test_on_unified_dataset(self.unified_test_loader, final_model)
            final_acc = final_test_res.get('accuracy', 0.0)
            class_stats = final_test_res.get('class_stats', {})
            print(f"Final Unified Test Accuracy: {final_acc:.4f}%")
            if self.global_num_classes > 0:
                random_guess_acc = (1.0 / self.global_num_classes) * 100
                print(f"Final Random Guess Acc:    {random_guess_acc:.4f}%")
            wandb.log({"final_unified_test_accuracy": final_acc})
            wandb.summary["final_unified_test_accuracy"] = final_acc

            print("\n--- Final Per-Class Performance ---")
            if class_stats:
                 for class_id in sorted(class_stats.keys()):
                     stats = class_stats[class_id]
                     correct, total = stats['correct'], stats['total']
                     class_acc = (correct / total) * 100 if total > 0 else 0.0
                     class_name = self.lab2cname.get(class_id, f"Unknown GID:{class_id}")
                     print(f"  Class {class_id:03d} ({class_name:<25}): {correct:>4} / {total:<4} ({class_acc:>6.2f}%) Correct")
            else: print("  No per-class statistics available.")
            print("-----------------------------------")

        else: print("Skipping final evaluation: Loader missing or weights invalid.")

        print("\n--- Saving Final Global Model ---")
        if self.check_weights_valid(self.global_weights):
            self.save_model(epoch=self.num_rounds, directory=self.cfg.OUTPUT_DIR, is_best=False,
                             val_result=final_test_res if 'final_test_res' in locals() else None) # Save final results
            print(f"Final model saved to {self.cfg.OUTPUT_DIR}")
            wandb.summary["final_model_saved"] = True
        else:
            print("Skipping final model save: Weights invalid.")
            wandb.summary["final_model_saved"] = False


    def save_model(self, epoch=None, directory="", is_best=False, val_result=None):
        # (Unchanged from previous version, but ensure val_result is passed if available)
        if not self.check_weights_valid(self.global_weights): print(f"Skipping save: weights invalid."); return
        if not directory: directory = self.cfg.OUTPUT_DIR; mkdir_if_missing(directory)
        subfolder = "FederatedAggregator"; target_dir = osp.join(directory, subfolder); mkdir_if_missing(target_dir)
        checkpoint = {"epoch": epoch if epoch is not None else self.num_rounds, "state_dict": self.global_weights, "optimizer": None, "scheduler": None, "val_result": val_result, "cfg": self.cfg.dump()}
        fpath = save_checkpoint(checkpoint, target_dir, is_best=is_best, model_name="aggregator", epoch=epoch)
        if self.cfg.VERBOSE: print(f"Global model saved to: {fpath}")
        # --- W&B Artifact Logging ---
        try:
             if osp.exists(fpath):
                 artifact_name = f"aggregator_model_round_{epoch}" if epoch else "aggregator_model_final";
                 if is_best: artifact_name += "_best"
                 accuracy_meta = val_result.get('accuracy', None) if val_result else None # Get accuracy safely
                 artifact = wandb.Artifact(name=artifact_name, type="model", metadata={"round": epoch, "is_best": is_best, "unified_test_accuracy": accuracy_meta})
                 artifact.add_file(fpath); wandb.log_artifact(artifact)
                 if self.cfg.VERBOSE: print(f"W&B artifact '{artifact_name}' logged from {fpath}")
             else: print(f"Warning: Model file {fpath} not found after save. Cannot log artifact.")
        except Exception as e: print(f"Error logging W&B artifact: {e}")


    def load_model(self, directory, epoch=None, expected_file_hash=None):
        # (Unchanged)
        if not directory: print("Skipping load: No dir provided."); return
        subfolder = "FederatedAggregator"; model_file = f"aggregator-model.pth.tar-{epoch}" if epoch else "aggregator-model.pth.tar"; path = osp.join(directory, subfolder, model_file)
        if not osp.exists(path): alt_path = osp.join(directory, subfolder, "aggregator-model.pth.tar"); path = alt_path if osp.exists(alt_path) else path # Check default name too
        if not osp.exists(path): raise FileNotFoundError(f"Aggregator model not found at {path}")
        print(f"Loading global model from: {path}")
        if expected_file_hash: file_hash = self.compute_file_hash(path); print(f"SHA-256: {file_hash}"); assert file_hash == expected_file_hash, "Hash mismatch!"
        ckpt = load_checkpoint(path); state_dict = ckpt.get("state_dict"); loaded_epoch = ckpt.get("epoch", None)
        if state_dict is None: raise ValueError(f"No 'state_dict' in {path}.")
        if self.check_weights_valid(state_dict): self.global_weights = state_dict; print(f"Loaded aggregator weights from epoch {loaded_epoch}.")
        else: raise ValueError("Loaded weights invalid (NaN/Inf)! Aborting.")


    # --- Debugging Helpers (Unchanged) ---
    def debug_print_samples(self, data_manager, subset="train_x", max_per_class=2): pass # Keep implementation if needed
    def debug_clients_data(self, max_clients_to_show=5): pass # Keep implementation if needed
    def debug_save_samples_images(self, data_manager, subset="train_x", output_dir="debug_samples", max_per_class=5): pass # Keep implementation if needed


    def test(self, evaluate_train=False):
        # (Calls enhanced test_on_unified_dataset)
        print("\n--- Running Final Test using Unified Test Loader ---")
        if self.unified_test_loader and self.check_weights_valid(self.global_weights):
            model_to_test = self.clients[0].model
            model_to_test.load_state_dict(self.global_weights)
            model_to_test.eval()
            results = self.test_on_unified_dataset(self.unified_test_loader, model_to_test) # Gets detailed results
            print(f"Final Unified Test Results: Accuracy = {results.get('accuracy', 0.0):.4f}%")
            # Optionally print detailed class stats here too like in finalize_training
            return results
        else: print("Cannot run final test: Loader missing or weights invalid."); return {"accuracy": 0.0, "class_stats": {}}