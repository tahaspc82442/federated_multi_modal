

from dassl.engine import TRAINER_REGISTRY, TrainerX
from trainers.maple import MaPLe
from dassl.data import DataManager
from .data_partition import partition_dataset_iid
from .client_datamanager import ClientDataManager
import torch
import numpy as np
from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
import torch.nn.functional as F 
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from torch.cuda.amp import GradScaler, autocast
import os.path as osp
from dassl.utils import (
    mkdir_if_missing, 
    load_pretrained_weights,
    load_checkpoint,
    save_checkpoint,
    tolist_if_not
)
import copy
@TRAINER_REGISTRY.register()
class MaPLeFederated(TrainerX):
    def __init__(self, cfg):
        self.cfg = cfg
        self.num_clients = cfg.FED.NUM_CLIENTS
        self.num_rounds = cfg.FED.NUM_ROUNDS
        self.local_epochs = cfg.FED.LOCAL_EPOCHS
        self.clients = []
        self.global_weights = None
        self.nan_stats = {
            'total_updates': 0,
            'failed_clients': [],
            'skipped_rounds': 0
        }
        super().__init__(cfg)

    def test(self):
        if self.check_weights_valid(self.global_weights):
            self.broadcast_weights(self.global_weights)
            self.clients[0].model.eval()
            with torch.no_grad():
                results = self.clients[0].test(evaluate_train =False)  # CHANGE BASED ON WHETHER RUNNING X_D_TEST OR X_D TRAIN



    def build_data_loader(self):
        """Build unified label space across UC Merced & PatternNet, aligning duplicates."""

        # 1) Load each dataset with your existing code
        patternnet_cfg = self.cfg.clone()
        patternnet_cfg.defrost()
        patternnet_cfg.DATASET.NAME = "PatternNet"
        dm_patternnet = DataManager(patternnet_cfg)
        dataset_patternnet = dm_patternnet.dataset

        ucmerced_cfg = self.cfg.clone()
        ucmerced_cfg.defrost()
        ucmerced_cfg.DATASET.NAME = "Ucmerced"
        dm_ucmerced = DataManager(ucmerced_cfg)
        dataset_ucmerced = dm_ucmerced.dataset

        # 2) Extract local label2class for each
        pn_lab2cname = dm_patternnet.lab2cname  # e.g. {0: "airplane", 1: "bridge", ...}
        uc_lab2cname = dm_ucmerced.lab2cname

        # --- UC Merced renaming logic here ---
        # Define any mappings from old to new names
        rename_map = {
    # UC Merced -> PatternNet (exact or closest equivalent)
            "tenniscourt": "tennis_court",
            "golfcourse": "golf_course",
            "parkinglot": "parking_lot",
            "storagetanks": "storage_tank",
            "mobilehomepark": "mobile_home_park",
            "baseballdiamond": "baseball_field",
            "denseresidential": "dense_residential",
            "sparseresidential": "sparse_residential"  }
        # Apply renaming
        for k, old_cname in uc_lab2cname.items():
            if old_cname in rename_map:
                uc_lab2cname[k] = rename_map[old_cname]

        # 3) Convert to sets of class names
        set_pn = set(pn_lab2cname.values())
        set_uc = set(uc_lab2cname.values())

        # 4) Merge them into one global set
        global_set = set_pn.union(set_uc)
        global_list = sorted(list(global_set))

        print("global_list", global_list)

        # 5) Build name->global_id, global_id->name
        name2gid = {}
        gid2name = {}
        for i, cname in enumerate(global_list):
            name2gid[cname] = i
            gid2name[i] = cname

        global_num_classes = len(global_list)
        print(f"[INFO] Unified #classes = {global_num_classes}")

        # Helper: invert a dict
        def invert_dict(d):
            return {v: k for k, v in d.items()}

        pn_name2lab = invert_dict(pn_lab2cname)
        uc_name2lab = invert_dict(uc_lab2cname)

        # 6) Remap datasets to global IDs
        def remap_dataset(data_list, name2gid, local_lab2cname):
            """Remap each sample's local label to the global index."""
            for idx, item in enumerate(data_list):
                local_label = item.label
                class_name = local_lab2cname[local_label]
                global_label = name2gid[class_name]

                # Construct a new Datum with the updated label
                new_item = Datum(
                    impath=item.impath,
                    label=global_label,
                    classname=class_name,  # or item.classname if you want to keep the old class name
                    caption=item.caption
                )

                # Replace the old Datum in the list
                data_list[idx] = new_item

        # Remap each split in PatternNet
        remap_dataset(dataset_patternnet.train_x, name2gid, pn_lab2cname)
        remap_dataset(dataset_patternnet.val,     name2gid, pn_lab2cname)
        remap_dataset(dataset_patternnet.test,    name2gid, pn_lab2cname)

        # Remap each split in UC Merced
        remap_dataset(dataset_ucmerced.train_x, name2gid, uc_lab2cname)
        remap_dataset(dataset_ucmerced.val,     name2gid, uc_lab2cname)
        remap_dataset(dataset_ucmerced.test,    name2gid, uc_lab2cname)

        # 7) Store final global label space in your federated trainer
        self.lab2cname = gid2name
        self.num_classes = global_num_classes

        # Also ensure your model sees the correct # of classes
        print("####")
        print(self.num_classes)
        self.cfg.defrost()
        self.cfg.MODEL.NUM_CLASSES = self.num_classes
        self.cfg.freeze()

        # 8) Build ClientDataManager for each client
        dm_client_0 = ClientDataManager(
            train_x=dataset_patternnet.train_x,
            val=dataset_patternnet.val,
            test=dataset_patternnet.test,
            cfg=self.cfg
        )

        dm_client_1 = ClientDataManager(
            train_x=dataset_ucmerced.train_x,
            val=dataset_ucmerced.val,
            test=dataset_ucmerced.test,
            cfg=self.cfg
        )

        self.client_data_managers = [dm_client_0, dm_client_1]

        # No top-level data loaders needed
        self.train_loader_x = None
        self.val_loader = None
        self.test_loader = None
        self.dm = None




        
    def build_data_loader_one(self):
        """
        2) The default parent's build_data_loader() is replaced.
           We load a single dataset, then partition it among multiple clients.
        """
        # Step 2.1: Build the "main" dataset
        dm_main = DataManager(self.cfg)
        dataset = dm_main.dataset  # has train_x, val, test
        self.lab2cname = dm_main.lab2cname
        self.num_classes = dm_main.num_classes

        # Step 2.2: Partition dataset for each client
        subsets = partition_dataset_iid(dataset, num_clients=self.num_clients)

        # Step 2.3: Build a ClientDataManager for each subset
        self.client_data_managers = []
        for (train_i, val_i, test_i) in subsets:
            dm_i = ClientDataManager(
                train_x=train_i,
                val=val_i,
                test=test_i,
                cfg=self.cfg
            )
            self.client_data_managers.append(dm_i)

        # The parent trainer tries to set train_loader_x, etc.
        # We don't need them for the top-level "federated" trainer
        self.train_loader_x = None
        self.val_loader = None
        self.test_loader = None
        self.dm = None
    
    def build_model(self):
        """
        3) Called by the parent. We create a local `MaPLe` trainer for each client.
        """
        self.clients = []
        for i, dm in enumerate(self.client_data_managers):
            # Create the single-site trainer
            client_trainer = MaPLe(self.cfg, client_id=i)
            client_trainer.dm = dm
            # Build model, optimizer, scheduler inside
            client_trainer.build_model()
            self.clients.append(client_trainer)

        # Initialize global weights from client 0's model
        self.global_weights = self.clients[0].model.state_dict()

    def train(self):
        for round_idx in range(self.num_rounds):
            print(f"\n--- Federated Round {round_idx+1}/{self.num_rounds} ---")
            
            # Broadcast with NaN check
            if self.check_weights_valid(self.global_weights):
                self.broadcast_weights(self.global_weights)
            else:
                print("Invalid global weights detected! Reverting to previous state.")
                self.nan_stats['skipped_rounds'] += 1
                continue

            local_state_dicts = []
            valid_clients = 0
            
            for i, client_trainer in enumerate(self.clients):
                print(f"[Client {i}] Local training ...")
                client_trainer.epoch = round_idx * self.local_epochs
                client_trainer.max_epoch = (round_idx + 1) * self.local_epochs

                # Run local training with gradient safety
                try:
                    for local_epoch in range(client_trainer.epoch, client_trainer.max_epoch):
                        client_trainer.run_epoch(local_epoch)
                except RuntimeError as e:
                    print(f"Client {i} failed training: {str(e)}")
                    self.nan_stats['failed_clients'].append(i)
                    continue

                # Collect weights with validation
                client_weights = client_trainer.model.state_dict()
                if self.check_weights_valid(client_weights):
                    local_state_dicts.append(client_weights)
                    valid_clients += 1
                else:
                    print(f"Client {i} produced invalid weights, skipping aggregation")
                    # Reset to global weights for next round
                    client_trainer.model.load_state_dict(self.global_weights)

            # Handle aggregation safely
            if valid_clients > 0:
                self.global_weights = self.safe_average_weights(local_state_dicts, valid_clients)
                self.nan_stats['total_updates'] += 1
            else:
                print("All clients failed! Keeping previous global model")
                self.nan_stats['skipped_rounds'] += 1

        self.finalize_training()

    def safe_average_weights(self, local_state_dicts, valid_clients):
        """Robust FedAvg with NaN protection"""
        avg_state = {}
        for key in local_state_dicts[0].keys():
            stacked = torch.stack([sd[key].float() for sd in local_state_dicts])
            
            # Replace NaNs with 0 and Infs with large finite values
            stacked = torch.nan_to_num(stacked, nan=0.0, posinf=1e4, neginf=-1e4)
            
            # Mean with error checking
            if torch.is_tensor(stacked):
                avg_state[key] = torch.mean(stacked, dim=0).half()
            else:  # Handle non-tensor parameters
                avg_state[key] = np.mean(stacked, axis=0)
        return avg_state

    def check_weights_valid(self, state_dict):
        """Comprehensive validity check for model weights"""
        for name, param in state_dict.items():
            if torch.isnan(param).any():
                print(f"NaN detected in {name}")
                return False
            if torch.isinf(param).any():
                print(f"Inf detected in {name}")
                return False
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"NaN gradient in {name}")
                return False
        return True

    def finalize_training(self):
        #Handle final model validation and stats
        print("\nTraining Summary:")
        print(f"Completed Rounds: {self.nan_stats['total_updates']}")
        print(f"Skipped Rounds: {self.nan_stats['skipped_rounds']}")
        print(f"Client Failure Rate: {len(self.nan_stats['failed_clients'])/self.num_clients:.1%}")
        
        # Test best valid model
        if self.check_weights_valid(self.global_weights):
            self.broadcast_weights(self.global_weights)
            self.clients[0].model.eval()
            with torch.no_grad():
                results = self.clients[0].test()
            #print(f"Final test accuracy: {results['accuracy']:.1%}")
            self.before_save()
            self.save_model()
            print(self.cfg.dump())
        else:
            print("Final global model invalid! Testing aborted.")
        

    
    
 
    def save_model(self, epoch=None, directory="", is_best=False, val_result=None):
        """
        Save checkpoint in a subfolder named after client_id (or 'aggregator' if None),
        and produce a clean file:  model.pth.tar
        """
        if not directory:
            directory = self.cfg.OUTPUT_DIR
        mkdir_if_missing(directory)

        # Decide subfolder name
        
        subfolder = f"MultiModalPromptLearner_{0}"

        # Create subfolder inside the main directory
        target_dir = osp.join(directory, subfolder)
        mkdir_if_missing(target_dir)

        # Prepare your Dassl-friendly checkpoint
        checkpoint = {
            "epoch": self.cfg.OPTIM.MAX_EPOCH,
            "state_dict": self.global_weights,
            "optimizer": None,  # or some actual optimizer state if needed
            "scheduler": None,  # or some actual scheduler state if needed
            "val_result": val_result,
            "cfg": self.cfg.dump()
        }

        # Final file: model.pth.tar (no extra `-2` unless there's a collision from repeated saves)
        filepath = target_dir

        # We do NOT pass model_name here; we provide the final path directly
        save_checkpoint(checkpoint, filepath, is_best=is_best)

        if self.cfg.VERBOSE:
            print(f"Model saved to {filepath}")

    def load_model(self, directory, epoch=None):
        """Load the global (aggregator) weights from a saved checkpoint."""
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        # The aggregator subdirectory you used in save_model()
        # e.g. "MultiModalPromptLearner_0" if aggregator is "client_id=0"
        subfolder = "MultiModalPromptLearner_0"

        # By default, let's load "model.pth.tar".
        # If you saved different epoch checkpoints as "model.pth.tar-<epoch>",
        # handle that logic here:
        if epoch is not None:
            model_file = f"model.pth.tar-{epoch}"
        else:
            model_file = "model.pth.tar"

        # Construct the full path
        model_path = osp.join(directory, subfolder, model_file)

        # Check existence
        if not osp.exists(model_path):
            raise FileNotFoundError(f"Model not found at '{model_path}'")

        # Load checkpoint via Dassl's utility
        checkpoint = load_checkpoint(model_path)
        state_dict = checkpoint["state_dict"]
        loaded_epoch = checkpoint.get("epoch", None)

        # Optionally remove token prefix/suffix if aggregator doesn't need them
        if "prompt_learner.token_prefix" in state_dict:
            del state_dict["prompt_learner.token_prefix"]
        if "prompt_learner.token_suffix" in state_dict:
            del state_dict["prompt_learner.token_suffix"]

        # Overwrite aggregator's global weights
        self.global_weights = state_dict

        print(f"Loaded global weights from '{model_path}' (epoch={loaded_epoch}).")

        # Optionally broadcast these newly loaded weights to clients immediately
        if self.check_weights_valid(self.global_weights):
            self.broadcast_weights(self.global_weights)
            print("Broadcasted loaded global weights to all clients.")
        else:
            print("Warning: loaded global weights are invalid! Skipping broadcast.")


    
    def before_save(self):
        """Sync global weights into registered models"""
        # Make sure base trainer saves the current global weights
        for name in self.get_model_names():
            self._models[name].load_state_dict(self.global_weights)
    def finalize_training2(self, print_samples=5):
        """Final evaluation with optional prediction printing"""
        if self.check_weights_valid(self.global_weights):
            print("\n--- Final Global Model Evaluation ---")
            self.broadcast_weights(self.global_weights)
            
            # Get first client's components
            client = self.clients[0]
            test_loader = client.dm.test_loader
            class_names = client.dm.lab2cname
            
            # Evaluation storage
            all_preds = []
            all_labels = []
            
            client.model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    inputs = batch["img"].to(client.device)
                    labels = batch["label"].to(client.device)
                    outputs = client.model(inputs)
                    
                    # Store results
                    all_preds.append(outputs.cpu())
                    all_labels.append(labels.cpu())
                    
                    # Print first batch samples if requested
                    if batch_idx == 0 and print_samples > 0:
                        self._print_sample_predictions(
                            outputs, 
                            labels,
                            class_names,
                            n_samples=print_samples
                        )

            # Calculate overall accuracy
            preds = torch.cat(all_preds)
            labels = torch.cat(all_labels)
            accuracy = (preds.argmax(dim=1) == labels).float().mean().item()
            print(f"\nFinal Test Accuracy: {accuracy:.2%}")
        else:
            print("Skipping evaluation: Invalid global weights")

    def _print_sample_predictions(self, outputs, labels, class_names, n_samples=5):
        """Helper to print sample predictions with class names"""
        probs = F.softmax(outputs, dim=1)
        pred_classes = outputs.argmax(dim=1)
        
        print("\nSample Predictions (first batch):")
        print(f"{'Image':<5} | {'True Class':<20} | {'Predicted Class':<20} | Confidence")
        print("-" * 65)
        
        for i in range(min(n_samples, outputs.shape[0])):
            true_idx = labels[i].item()
            pred_idx = pred_classes[i].item()
            confidence = probs[i][pred_idx].item()
            
            true_name = class_names.get(true_idx, f"Class_{true_idx}")
            pred_name = class_names.get(pred_idx, f"Class_{pred_idx}")
            
            print(f"{i+1:<5} | {true_name:<20} | {pred_name:<20} | {confidence:.1%}")


    def broadcast_weights(self, global_state_dict):
        """Secure weight broadcasting with momentum reset"""
        for client_trainer in self.clients:
            # Load state dict with strict=False to handle partial mismatches
            client_trainer.model.load_state_dict(global_state_dict, strict=False)
            
            # Reset optimizer state safely
            for param_group in client_trainer.optim.param_groups:
                for param in param_group['params']:
                    if param in client_trainer.optim.state:
                        del client_trainer.optim.state[param]
            client_trainer.optim.param_groups[0]['params'] = [
                p for p in client_trainer.model.parameters() if p.requires_grad
            ]

            # Rebuild scheduler without epoch argument
            client_trainer.sched = build_lr_scheduler(
                client_trainer.optim,
                client_trainer.cfg.OPTIM
            )
            
            # Manually set scheduler state if needed
            if hasattr(client_trainer, 'epoch'):
                # Set to last completed epoch
                client_trainer.sched.last_epoch = client_trainer.epoch - 1
    def _transfer_momentum_buffers(self, client_trainer):
        """Transfer momentum buffers from previous optimizer state"""
        new_optim = client_trainer.optim
        old_optim = client_trainer._prev_optim  # Store previous optimizer
        
        if old_optim is not None:
            for new_group, old_group in zip(new_optim.param_groups, old_optim.param_groups):
                for new_p, old_p in zip(new_group['params'], old_group['params']):
                    if 'momentum_buffer' in old_optim.state[old_p]:
                        new_optim.state[new_p]['momentum_buffer'] = \
                            old_optim.state[old_p]['momentum_buffer'].clone()
        
        client_trainer._prev_optim = new_optim  # Store for next round
            
    def check_model_weights(self,state_dict, tag=""):
        for name, param in state_dict.items():
            if torch.isnan(param).any():
                print(f"[DEBUG] NaN detected in weights: {name} {tag}")
            if torch.isinf(param).any():
                print(f"[DEBUG] Inf detected in weights: {name} {tag}")

# ... (keep other methods like build_data_loader, build_model, average_weights)