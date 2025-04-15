
"""Maple Fed original trainer
"""


import os.path as osp
import torch
import numpy as np
import torch.nn.functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.data import DataManager
from dassl.data.datasets import Datum
from dassl.utils import (
    mkdir_if_missing, 
    load_checkpoint,
    save_checkpoint
)
from dassl.optim import build_lr_scheduler
from collections import defaultdict
from PIL import Image
# Local single-site trainer
from trainers.maple import MaPLe
# Custom client data manager
from .client_datamanager import ClientDataManager
import os
from tqdm import trange
from .debug import debug_collate
import wandb
import os.path as osp
from trainers.maple import load_clip_to_cpu
import os.path as osp
import hashlib
from collections import Counter
import copy

@TRAINER_REGISTRY.register()
class MaPLeFederated(TrainerX):
    def __init__(self, cfg):
        # Must define self.lab2cname before super().__init__(cfg), 
        # because Dassl might build an evaluator in TrainerX.__init__
        self.lab2cname = {}
        self._clip_model = load_clip_to_cpu(cfg)
        if cfg.TRAINER.MAPLE.PREC == "fp16":
            self._clip_model = self._clip_model.half() 
        self._clip_model = self._clip_model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.cfg = cfg
        self.num_clients = cfg.FED.NUM_CLIENTS  # e.g. 2
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
        super().__init__(cfg)  # calls self.build_data_loader()

    ###################################################
    # A) Build the unified data loader
    ###################################################
    def build_data_loader(self):
        print("build data loader called")
        """
        1) Load PatternNet & UcMerced
        2) Rename classes in UcMerced so they match (if needed)
        3) Merge classes into a single global list
        4) Remap local labels to global IDs
        5) Set cfg.MODEL.NUM_CLASSES
        6) Create ClientDataManager for each
        """

        # -- 1) Load PatternNet
        pat_cfg = self.cfg.clone()
        pat_cfg.defrost()
        pat_cfg.DATASET.NAME = "PatternNet"
        dm_pn = DataManager(pat_cfg)
        dataset_pn = dm_pn.dataset
        self.dataset_pn = dataset_pn

        # -- 2) Load UcMerced
        uc_cfg = self.cfg.clone()
        uc_cfg.defrost()
        uc_cfg.DATASET.NAME = "Ucmerced"
        dm_uc = DataManager(uc_cfg)
        dataset_uc = dm_uc.dataset
        self.dataset_uc = dataset_uc

        # -- 3) Load EuroSAT
        euro_cfg = self.cfg.clone()
        euro_cfg.defrost()
        euro_cfg.DATASET.NAME = "EuroSAT"
        dm_euro = DataManager(euro_cfg)
        dataset_euro = dm_euro.dataset
        self.dataset_euro = dataset_euro    

        # --4 ) Load Mlrs
        mlrs_cfg=self.cfg.clone()
        mlrs_cfg.defrost()
        mlrs_cfg.DATASET.NAME="Mlrs"
        dm_mlrs=DataManager(mlrs_cfg)
        dataset_mlrs=dm_mlrs.dataset
        self.dataset_mlrs=dataset_mlrs

        # -- 5) Load Milaid
        milaid_cfg=self.cfg.clone()
        milaid_cfg.defrost()
        milaid_cfg.DATASET.NAME="Milaid"
        dm_milaid=DataManager(milaid_cfg)
        dataset_milaid=dm_milaid.dataset
        self.dataset_milaid=dataset_milaid

        # local label->classname
        pn_lab2cname = dm_pn.lab2cname
        uc_lab2cname = dm_uc.lab2cname
        euro_lab2cname = dm_euro.lab2cname
        mlrs_lab2cname=dm_mlrs.lab2cname
        milaid_lab2cname=dm_milaid.lab2cname

        rename_map = {
            "tenniscourt": "tennis_court",
            "golfcourse": "golf_course",
            "parkinglot": "parking_lot",
            "storagetanks": "storage_tank",
            "mobilehomepark": "mobile_home_park",
            "baseballdiamond": "baseball_field",
            "denseresidential": "dense_residential",
            "sparseresidential": "sparse_residential"
        }
        for k, old_cname in uc_lab2cname.items():
            if old_cname in rename_map:
                uc_lab2cname[k] = rename_map[old_cname]

        milaid_rename_map = {
        # Fix spaces -> underscores
        "commercial area": "commercial_area",
        "ice land": "ice_land",
        "bare land": "bare_land",
        "detached house": "detached_house",
        "dry field": "dry_field",
        "golf course": "golf_course",
        "ground track field": "ground_track_field",
        "mobile home park": "mobile_home_park",
        "oil field": "oil_field",
        "paddy field": "paddy_field",
        "parking lot": "parking_lot",
        "rock land": "rock_land",
        "solar power plant": "solar_power_plant",
        "sparse shrub land": "sparse_shrub_land",
        "storage tank": "storage_tank",
        "swimming pool": "swimming_pool",
        "terraced field": "terraced_field",
        "train station": "train_station",
        "wastewater plant": "wastewater_plant",
        "wind turbine": "wind_turbine",
        
        # Add these if they exist in your data
        "baseball field": "baseball_field",  # From your earlier example
        "basketball court": "basketball_court",
        "tennis court": "tennis_court",       # Fix typo if needed

}       
        
        # Apply to MILAID's lab2cname dictionary
        for label, old_cname in milaid_lab2cname.items():
            if old_cname in milaid_rename_map:
                milaid_lab2cname[label] = milaid_rename_map[old_cname]

        # -- 3) Form global list of classes (union)
        set_pn = set([cname.lower() for cname in pn_lab2cname.values()])
        set_uc = set([cname.lower() for cname in uc_lab2cname.values()])
        set_euro = set([cname.lower() for cname in euro_lab2cname.values()])
        set_mlrs = set([cname.lower() for cname in mlrs_lab2cname.values()])
        set_milaid = set([cname.lower() for cname in milaid_lab2cname.values()])
        global_list = sorted(list(set_pn.union(set_uc).union(set_euro).union(set_mlrs).union(set_milaid)))
        #global_list = sorted(list(set_pn.union(set_uc).union(set_euro)))
        global_num_classes = len(global_list)
        print(f"[INFO] Unified #classes = {global_num_classes}")


        _datasets = {
            "pn": pn_lab2cname,
            "uc": uc_lab2cname,
            "euro": euro_lab2cname,
            "mlrs": mlrs_lab2cname,
            "milaid": milaid_lab2cname
                }

# Print label-to-classname mappings for each dataset
        for dataset_name, lab2cname in _datasets.items():
            print(f"\n{dataset_name.upper()} Label-to-Class Mappings:")
            for label in sorted(lab2cname.keys()):
                print(f"Label {label}: {lab2cname[label]}")

        # Build name->gid
        name2gid = {cname: i for i, cname in enumerate(global_list)}

        # Save a dictionary {class_id -> class_name} for Dassl
        self.lab2cname = {i: cname for i, cname in enumerate(global_list)}


        print("*************************")
        print(self.lab2cname)


        for dataset_name, lab2cname in _datasets.items():
            for label, classname in lab2cname.items():
                if classname not in global_list:
                    print(f"ERROR: {classname} (from {dataset_name}) not in global_list!")


        for i, datum in enumerate(dataset_mlrs.test):
            try:
                with Image.open(datum.impath) as im:
                    im.verify()
            except Exception as e:
                print(f"Corrupt file: {datum.impath}\n{e}")

        for i, datum in enumerate(dataset_mlrs.train_x):
            try:
                with Image.open(datum.impath) as im:
                    im.verify()
            except Exception as e:
                print(f"Corrupt file: {datum.impath}\n{e}")
                
        for i, datum in enumerate(dataset_mlrs.val):
            try:
                with Image.open(datum.impath) as im:
                    im.verify()
            except Exception as e:
                print(f"Corrupt file: {datum.impath}\n{e}")


        print("MLRS train size:", len(dataset_mlrs.train_x))
        print("MLRS val size:", len(dataset_mlrs.val))
        print("MLRS test size:", len(dataset_mlrs.test))

        # -- 4) Remap local labels -> global IDs
        def remap(data_list, local_lab2cname):
            for idx, item in enumerate(data_list):
                old_label = item.label
                cname = local_lab2cname[old_label].lower()
                gid = name2gid[cname]
                data_list[idx] = Datum(
                    impath=item.impath,
                    label=gid,
                    classname=cname,
                    caption=item.caption
                )

        remap(dataset_pn.train_x, pn_lab2cname)
        remap(dataset_pn.val,     pn_lab2cname)
        remap(dataset_pn.test,    pn_lab2cname)

        remap(dataset_uc.train_x, uc_lab2cname)
        remap(dataset_uc.val,     uc_lab2cname)
        remap(dataset_uc.test,    uc_lab2cname)

        remap(dataset_euro.train_x, dm_euro.lab2cname)
        remap(dataset_euro.val, dm_euro.lab2cname)
        remap(dataset_euro.test, dm_euro.lab2cname)

        remap(dataset_mlrs.train_x, dm_mlrs.lab2cname)
        remap(dataset_mlrs.val, dm_mlrs.lab2cname)
        remap(dataset_mlrs.test, dm_mlrs.lab2cname)

        remap(dataset_milaid.train_x, dm_milaid.lab2cname)
        remap(dataset_milaid.val, dm_milaid.lab2cname)
        remap(dataset_milaid.test, dm_milaid.lab2cname)

        # -- 5) Overwrite cfg.MODEL.NUM_CLASSES
        self.cfg.defrost()
        self.cfg.MODEL.NUM_CLASSES = global_num_classes
        self.cfg.freeze()

        # -- 6) Create ClientDataManager
        dm_client_0 = ClientDataManager(
            train_x=dataset_pn.train_x,
            val=dataset_pn.val,
            test=dataset_pn.test,
            cfg=self.cfg
        )
        dm_client_1 = ClientDataManager(
            train_x=dataset_uc.train_x,
            val=dataset_uc.val,
            test=dataset_uc.test,
            cfg=self.cfg
        )

        dm_client_2 = ClientDataManager(
        train_x=dataset_euro.train_x,
        val=dataset_euro.val,
        test=dataset_euro.test,
        cfg=self.cfg
                                        )
        dm_client_3 = ClientDataManager(
            train_x=dataset_mlrs.train_x,
            val=dataset_mlrs.val,
            test=dataset_mlrs.test,
            cfg=self.cfg,
        )
        dm_client_4 = ClientDataManager(
            train_x=dataset_milaid.train_x,
            val=dataset_milaid.val,
            test=dataset_milaid.test,
            cfg=self.cfg
        )

        self.client_data_managers = [dm_client_0, dm_client_1,dm_client_2,dm_client_3,dm_client_4]




        dm_client_3.train_loader.collate_fn = debug_collate
        dm_client_2.train_loader.collate_fn = debug_collate



        train_list = dm_client_3.train_x_list  # The raw list of Datum objects
        print("Length of MLRS train list:", len(train_list))

        for idx, datum in enumerate(train_list):
            # 1) Check that the Datum itself is valid
            if datum is None:
                print(f"Got a None Datum at index {idx}")
                break
            
            # 2) Try to apply the same transforms that build_data_loader would apply
            try:
                img = Image.open(datum.impath).convert("RGB")
                img = dm_client_3.tfm_train(img)  # run the training transforms
                # If needed, you could also check shape: print(img.shape)
            except Exception as e:
                print(f"Error applying transform at index={idx}, path={datum.impath}\n{e}")
                break

        from collections import Counter
        train_labels = [d.label for d in dm_client_3.train_x_list]
        count_by_label = Counter(train_labels)
        print("Label frequencies:", count_by_label)

        self.train_loader_x = None
        self.val_loader = None
        self.test_loader = None
        self.dm = None
        #self.debug_clients_data() 
        print("build data loader done, we have all the clients data")   


    def create_unified_test_dataloader(self):
        """
        Create a unified test set by combining test sets from all five datasets:
        PatternNet, UcMerced, EuroSAT, Mlrs, and Milaid.
        
        Returns:
            A DataManager instance with the unified test set
        """
        # Collect individual test sets
        test_sets = [
            self.dataset_pn.test,
            self.dataset_uc.test,
            self.dataset_euro.test,
            self.dataset_mlrs.test, 
            self.dataset_milaid.test
        ]
        
        # Create unified test set
        unified_test = []
        dataset_sources = ["PatternNet", "UcMerced", "EuroSAT", "Mlrs", "Milaid"]
        dataset_counts = {ds: 0 for ds in dataset_sources}
        class_counts = {}
        
        # Combine all test sets and track statistics
        for dataset_name, test_set in zip(dataset_sources, test_sets):
            print(f"Adding {len(test_set)} samples from {dataset_name} test set")
            dataset_counts[dataset_name] = len(test_set)
            
            for item in test_set:
                unified_test.append(item)
                
                # Track class distribution
                if item.classname not in class_counts:
                    class_counts[item.classname] = 0
                class_counts[item.classname] += 1
        
        # Sanity checks
        print("\n=== Unified Test Set Sanity Checks ===")
        
        # 1. Overall statistics
        total_samples = len(unified_test)
        print(f"Total test samples: {total_samples}")
        print(f"Samples per dataset: {dataset_counts}")
        
        # 2. Check for path existence and image validity
        print("\nChecking for corrupt or missing files...")
        corrupt_files = []
        missing_files = []
        
        for i, datum in enumerate(unified_test):
            if i % 1000 == 0 and i > 0:
                print(f"Checked {i}/{total_samples} files...")
                
            # Check if file exists
            if not os.path.exists(datum.impath):
                missing_files.append(datum.impath)
                continue
                
            # Check if image is valid (sample check, comment out if too slow)
            if i % 100 == 0:  # Only check every 100th image to speed up
                try:
                    with Image.open(datum.impath) as img:
                        img.verify()
                except Exception as e:
                    corrupt_files.append((datum.impath, str(e)))
        
        if missing_files:
            print(f"WARNING: {len(missing_files)} missing files found")
            print(f"First 5 missing files: {missing_files[:5]}")
        else:
            print("All files exist ✓")
            
        if corrupt_files:
            print(f"WARNING: {len(corrupt_files)} corrupt files found")
            print(f"First 5 corrupt files: {corrupt_files[:5]}")
        else:
            print("All checked files valid ✓")
        
        # 3. Check class distribution
        print("\nClass distribution:")
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"Total classes: {len(sorted_classes)}")
        print("Top 10 classes by sample count:")
        for classname, count in sorted_classes[:10]:
            print(f"  {classname}: {count} samples")
        
        # 4. Check for label consistency
        print("\nChecking label consistency...")
        label_issues = []
        for datum in unified_test:
            if datum.label >= len(self.lab2cname) or datum.label < 0:
                label_issues.append((datum.impath, datum.label, datum.classname))
        
        if label_issues:
            print(f"WARNING: {len(label_issues)} label consistency issues found")
            print(f"First 5 label issues: {label_issues[:5]}")
        else:
            print("All labels within valid range ✓")
        
        # Create a data manager with the unified test set
        # Create empty train and val sets (we only need test)
        empty_train = []
        empty_val = []
        
        # Create a DataManager instance with our unified test set
        unified_data_manager = ClientDataManager(
                                train_x=empty_train,
                                val=empty_val,
                                test=unified_test,
                                cfg=self.cfg,
                                custom_tfm_train=None,
                                custom_tfm_test=None,
                                dataset_wrapper=None
                                                        )                       

        
        print(f"\nUnified test data loader created with {len(unified_test)} samples")
        
        return unified_data_manager




    ###################################################
    # B) Build local trainers (MaPLe)
    ###################################################

    
    def build_model(self):
        self.clients = []
        # We'll keep the global classnames in a variable to pass along
        global_classnames = list(self.lab2cname.values())

        for i, dm in enumerate(self.client_data_managers):
            local_trainer = MaPLe(self.cfg, client_id=i, classnames=global_classnames, _clip_model=self._clip_model)
            local_trainer.dm = dm
            local_trainer.build_model()
            self.clients.append(local_trainer)

        # initialize global weights from client 0
        self.global_weights = self.clients[0].model.state_dict()

    ###################################################
    # C) Federated training loop
    ###################################################
   
    def train(self):
          # minimal approach: local import so we can call wandb.log

        for round_idx in trange(self.num_rounds):
            print(f"\n--- Federated Round {round_idx+1}/{self.num_rounds} ---")

            # 1) Broadcast global weights (if valid)
            if self.check_weights_valid(self.global_weights):
                self.broadcast_weights(self.global_weights)
            else:
                print("Invalid global weights detected! Skipping round.")
                self.nan_stats["skipped_rounds"] += 1
                continue

            local_state_dicts = []
            valid_clients = 0
            valid_clients_list=[]
            
            # For logging local training losses
            round_losses = []

            # 2) Each client trains locally
            for i, trainer in enumerate(self.clients):
                print(f"[Client {i}] local training ...")
                trainer.epoch = round_idx * self.local_epochs
                trainer.max_epoch = (round_idx + 1) * self.local_epochs

                # We'll store the final epoch's average loss for this client
                last_epoch_loss = 0.0

                try:
                    # Run local epochs
                    for ep in range(trainer.epoch, trainer.max_epoch):
                        # run_epoch should return something like {"avg_loss": ...}
                        epoch_res = trainer.run_epoch(ep)
                        last_epoch_loss = epoch_res.get("avg_loss", 0.0)

                except RuntimeError as e:
                    print(f"Client {i} failed training: {str(e)}")
                    self.nan_stats["failed_clients"].append(i)
                    continue

                # Keep track of final local training loss after the last epoch
                round_losses.append(last_epoch_loss)

                #Log the last epoch loss per client (optional, but often helpful)
                wandb.log({
                    "round": round_idx,
                    f"client_{i}_local_loss": last_epoch_loss
                })

                # Collect weights
                w = trainer.model.state_dict()
                if self.check_weights_valid(w):
                    local_state_dicts.append(w)
                    valid_clients += 1
                    valid_clients_list.append(trainer)
                else:
                    print(f"Client {i} produced invalid weights, skipping aggregation")
                    trainer.model.load_state_dict(self.global_weights)

            # 3) Print average local training loss for this round
            if round_losses:
                avg_loss_this_round = sum(round_losses) / len(round_losses)
                print(f"[Round {round_idx+1}] Avg local training loss = {avg_loss_this_round:.4f}")
                # Log the round-level average loss
                wandb.log({
                    "round": round_idx,
                    "avg_loss_across_clients": avg_loss_this_round
                })

    

            # 4) Perform FedAvg if possible
            if valid_clients > 0:
                self.global_weights = self.safe_average_weights(local_state_dicts, valid_clients_list)
                self.nan_stats['total_updates'] += 1
            else:
                print("All clients failed! Reverting to previous global model.")
                self.nan_stats['skipped_rounds'] += 1

            # 5) (Optional) Evaluate test accuracy after each round using client 0
            if self.check_weights_valid(self.global_weights):
                self.broadcast_weights(self.global_weights)
                # Evaluate on client 0's test set (or loop all clients if desired)

                if (round_idx+1)%5==0:
                    self.save_model(epoch=round_idx+1)  # Save model every 5 rounds
                test_res_0 = self.clients[0].test()
                test_res_1 = self.clients[1].test()
                test_res_2 = self.clients[2].test()
                test_res_3 = self.clients[3].test()
                test_res_4 = self.clients[4].test()


                unified_test_manager = self.create_unified_test_dataloader()
                unified_test_res = self.test_on_unified_dataset(unified_test_manager.test_loader)
                print(f"=== Round {round_idx+1} Unified Test Results ===")
                print(f"Accuracy: {unified_test_res['accuracy']:.4f}")

                wandb.log({
                    "round": round_idx,
                    "test_accuracy_unified": unified_test_res["accuracy"]
                })


                if "accuracy" in test_res_0:
                    acc_val_0 = test_res_0["accuracy"]
                    acc_val_1 = test_res_1["accuracy"]
                    acc_val_2 = test_res_2["accuracy"]
                    acc_val_3 = test_res_3["accuracy"]
                    acc_val_4 = test_res_4["accuracy"]
                    print(f"[Round {round_idx+1}] Test accuracy (client 0) = {acc_val_0:.2f}%")
                    print(f"[Round {round_idx+1}] Test accuracy (client 1) = {acc_val_1:.2f}%")
                    print(f"[Round {round_idx+1}] Test accuracy (client 2) = {acc_val_2:.2f}%")
                    print(f"[Round {round_idx+1}] Test accuracy (client 3) = {acc_val_3:.2f}%")
                    print(f"[Round {round_idx+1}] Test accuracy (client 4) = {acc_val_4:.2f}%")
                    wandb.log({ 
                        "round": round_idx,
                        "test_accuracy_client_0": acc_val_0,
                        "test_accuracy_client_1": acc_val_1,
                        "test_accuracy_client_2": acc_val_2,
                        "test_accuracy_client_3": acc_val_3,
                        "test_accuracy_client_4": acc_val_4
                    })
                 
            else:    
                print("Global weights invalid after aggregation, skipping test.")

        # 6) Done training
        self.finalize_training()



    def test_on_unified_dataset(self, test_loader):
        """
        Test the current model on the unified test dataset
        
        Args:
            test_loader: DataLoader for the unified test set
            
        Returns:
            Dictionary with test metrics (accuracy, loss, etc.)
        """
        # Set model to evaluation mode
        
        # Initialize metrics
        total_correct = 0
        total_loss = 0
        total_samples = 0
        
        # Disable gradient computation for evaluation
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                # Move data to the appropriate device
                images, labels = batch['img'].to(self.device), batch['label'].to(self.device)
                
                # Forward pass
                self.clients[0].model.eval()
                outputs =self.clients[0].model(images)
                #loss = self.criterion(outputs, labels)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct = (predicted == labels).sum().item()
                
                # Update metrics
                total_correct += correct
                #total_loss += loss.item() * labels.size(0)
                total_samples += labels.size(0)
                
                # Optional: print progress
                if batch_idx % 20 == 0:
                    print(f"Testing batch {batch_idx}/{len(test_loader)}, "
                        f"Acc so far: {total_correct/total_samples:.4f}")
        
        # Calculate final metrics
        accuracy = (total_correct / total_samples)*100
        #avg_loss = total_loss / total_samples
        
        # Return metrics as dictionary
        return {
            "accuracy": accuracy,
            "total_samples": total_samples
        }

 

    def safe_average_weights(self, local_dicts, valid_clients):
        """Diversity-weighted federated averaging using class entropy"""
        # Calculate diversity scores for each valid client
        diversity_scores = [self._calculate_diversity(client) for client in valid_clients]
        total_score = sum(diversity_scores)
        
        # Handle zero-sum edge case
        if total_score == 0:
            total_score = 1e-8
            diversity_scores = [1/len(valid_clients)]*len(valid_clients)  # Fallback to uniform
        
        avg_state = {}
        for key in local_dicts[0].keys():
            # Weighted average using diversity scores
            weighted_tensors = [
                sd[key].float() * (score/total_score)
                for sd, score in zip(local_dicts, diversity_scores)
            ]
            
            # Stack and average with numerical stability
            stacked = torch.stack(weighted_tensors)
            stacked = torch.nan_to_num(stacked, nan=0.0, posinf=1e4, neginf=-1e4)
            
            avg_state[key] = torch.sum(stacked, dim=0).half()
        
        return avg_state

    def _calculate_diversity(self, client):
        """Calculate normalized entropy diversity score for a client"""
        # Get class distribution
        class_counts = Counter([d.classname for d in client.dm.train_x_list])
        
        # Convert to probabilities with smoothing
        counts = torch.tensor(list(class_counts.values()), dtype=torch.float32)
        probabilities = (counts + 1e-8) / (counts.sum() + 1e-8)  # Add epsilon to avoid NaN
        
        # Calculate normalized entropy
        entropy = -torch.sum(probabilities * torch.log(probabilities))
        max_entropy = torch.log(torch.tensor(len(class_counts), dtype=torch.float32))
        normalized_entropy = entropy / max_entropy
        
        # Final diversity score (higher = more diverse)
        return normalized_entropy.item()


    def check_weights_valid(self, state_dict):
        for name, param in state_dict.items():
            if torch.isnan(param).any():
                print(f"NaN in {name}")
                return False
            if torch.isinf(param).any():
                print(f"Inf in {name}")
                return False
        return True



    def compute_file_hash(self, path):
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

    def compute_state_dict_hash(self, state_dict):
        """Compute SHA-256 hash of the state_dict."""
        sha256 = hashlib.sha256()
        for key in sorted(state_dict.keys()):
            tensor = state_dict[key].cpu().numpy()  # Ensure tensor is on CPU and converted to numpy
            sha256.update(tensor.tobytes())
        return sha256.hexdigest()
    

    def broadcast_weights(self, global_sd):
        """Broadcast to each client with strict=True, ensuring same shape."""
        for client_trainer in self.clients:
            client_trainer.model.load_state_dict(global_sd, strict=False)
            # reset optimizer states
            for param_group in client_trainer.optim.param_groups:
                for p in param_group["params"]:
                    if p in client_trainer.optim.state:
                        del client_trainer.optim.state[p]
            # rebuild scheduler
            client_trainer.sched = build_lr_scheduler(client_trainer.optim, client_trainer.cfg.OPTIM)
            if hasattr(client_trainer, "epoch"):
                client_trainer.sched.last_epoch = client_trainer.epoch - 1
    # Inside MaPLeFederated class
    # fed prox is here
    # def broadcast_weights(self, global_sd):
    #     """Broadcast to each client and store initial global weights for FedProx."""
    #     global_sd_cpu = {k: v.cpu() for k, v in global_sd.items()} # Keep a CPU copy for storage

    #     for client_trainer in self.clients:
    #         # Load the current global state into the client model
    #         # Make sure keys match, use strict=False if needed but verify compatibility
    #         client_trainer.model.load_state_dict(global_sd, strict=False)

    #         # Store a deep copy of the initial global weights (on CPU) for FedProx calculation later
    #         if self.prox_mu > 0:
    #             # Deepcopy ensures we don't just store a reference
    #             client_trainer.global_model_params_start_round = copy.deepcopy(global_sd_cpu)
    #         else:
    #             client_trainer.global_model_params_start_round = None # Not needed if mu is 0

    #         # Reset optimizer states (important!)
    #         # If using Adam, SGD with momentum etc., state needs reset when model changes
    #         client_trainer.optim.state = defaultdict(dict) # Resets optimizer state

    #         # Rebuild scheduler if necessary (or reset its state)
    #         client_trainer.sched = build_lr_scheduler(client_trainer.optim, client_trainer.cfg.OPTIM)
    #         # Ensure scheduler epoch matches the current training phase
    #         if hasattr(client_trainer, "epoch"):
    #             # Dassl schedulers often step based on epoch, reset internal counter
    #             client_trainer.sched.last_epoch = client_trainer.epoch - 1 # Adjust as needed based on scheduler type


    def finalize_training(self):
        print("\nTraining Summary:")
        print(f"Completed Rounds: {self.nan_stats['total_updates']}")
        print(f"Skipped Rounds: {self.nan_stats['skipped_rounds']}")
        fail_rate = len(self.nan_stats['failed_clients']) / max(1, self.num_clients)
        print(f"Client Failure Rate: {fail_rate:.1%}")

        # Evaluate final global
        if self.check_weights_valid(self.global_weights):
            self.broadcast_weights(self.global_weights)
            self.clients[0].model.eval()
            with torch.no_grad():
                result = self.clients[0].test()
            print("Final test result:", result)

            # optional: save final model
            self.before_save()
            self.save_model()
        else:
            print("Final global invalid, no test.")

    def before_save(self):
        """Sync aggregator's global weights into base trainer for Dassl saving."""
        for name in self.get_model_names():
            self._models[name].load_state_dict(self.global_weights)

    





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








    def test_on_all_clients(self):
        """
        Test the current model on all clients' test sets
        """
        for i, trainer in enumerate(self.clients):
            print(f"\n--- Testing on Client {i} ---")
            trainer.model.eval()
            with torch.no_grad():
                result = trainer.test()
            print(f"Test result on dataset of Client {i}:", result)
            wandb.log({
                "model accuracy using client dataset": i,
                "test_accuracy": result['accuracy']
            })
        return

  


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


    def debug_print_samples(self,data_manager, subset="train_x", max_per_class=5):
        """
        Print up to `max_per_class` samples for each class in the given subset.
        Each sample includes label, class name, caption, image path, etc.
        """
        

        # Access the chosen subset from the DM's dataset
        data_subset = getattr(data_manager.dataset, subset, None)
        if not data_subset:
            print(f"No data found for subset='{subset}'!")
            return

        # Group samples by (label or classname)
        class_dict = defaultdict(list)
        for d in data_subset:
            class_dict[d.classname].append(d)

        print(f"\n--- Debugging {subset.upper()} ---")
        for cname, samples in class_dict.items():
            print(f"\nClass '{cname}' ({len(samples)} samples)")
            for i, datum in enumerate(samples[:max_per_class]):
                print(
                    f"  Sample {i+1} | "
                    f"label={datum.label}, "
                    f"caption='{datum.caption}', "
                    f"impath='{datum.impath}'"
                )
        print("--- End of Debug ---\n")



    
    def debug_clients_data(self):
        

        
        client_names = ["pn", "uc", "euro", "mlrs", "milaid"]
        
        for client_idx, (client_name, client_dm) in enumerate(zip(client_names, self.client_data_managers)):
            print(f"\n=== Client {client_idx}: {client_name.upper()} ===")
            
            # Collect all splits
            splits = {
            "train": client_dm.train_x_list,  # Use .train_x_list instead of .train_x
            "val": client_dm.val_list,        # .val_list
            "test": client_dm.test_list       # .test_list
             }
            
            for split_name, split_data in splits.items():
                print(f"\n  Split: {split_name.upper()}")
                
                # Get class names via global lab2cname (to ensure lowercase consistency)
                class_names = [self.lab2cname[datum.label] for datum in split_data]
                
                # Count class frequencies
                from collections import Counter
                counter = Counter(class_names)
                
                # Print sorted by class name
                for cls in sorted(counter.keys()):
                    print(f"    - {cls}: {counter[cls]} samples")
                
                # Print totals
                print(f"    Total classes: {len(counter)}")
                print(f"    Total images: {len(split_data)}")

    def debug_save_samples_images(self,data_manager, subset="train_x",
                              output_dir="debug_samples",
                              max_per_class=5):
        """
        Copy/save up to `max_per_class` images per class from the given subset
        into a directory structure:
            debug_samples/<subset>/<classname>/<sample_X_label_Y.jpg>
        """
        data_subset = getattr(data_manager.dataset, subset, None)
        if not data_subset:
            print(f"No data found for subset='{subset}'!")
            return

        # Group samples by class name
        class_dict = defaultdict(list)
        for d in data_subset:
            class_dict[d.classname].append(d)

        # Prepare an output subfolder for this subset
        os.makedirs(output_dir, exist_ok=True)
        subset_dir = os.path.join(output_dir, subset)
        os.makedirs(subset_dir, exist_ok=True)

        # For each class, copy up to max_per_class images
        for cname, samples in class_dict.items():
            class_dir = os.path.join(subset_dir, cname)
            os.makedirs(class_dir, exist_ok=True)

            for i, datum in enumerate(samples[:max_per_class]):
                # Load the image
                img_pil = Image.open(datum.impath).convert("RGB")

                # Construct a filename
                # e.g. "sample_1_label_0.jpg"
                save_name = f"sample_{i+1}_label_{datum.label}.jpg"
                save_path = os.path.join(class_dir, save_name)

                # Save/copy the image
                img_pil.save(save_path)

        print(f"Saved up to {max_per_class} images per class to: {subset_dir}")

    def test(self):
        if self.check_weights_valid(self.global_weights):
            self.broadcast_weights(self.global_weights)
            self.clients[0].model.eval()
            with torch.no_grad():
                results = self.clients[0].test(evaluate_train =True)


