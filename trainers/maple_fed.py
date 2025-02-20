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

@TRAINER_REGISTRY.register()
class MaPLeFederated(TrainerX):
    def __init__(self, cfg):
        # Must define self.lab2cname before super().__init__(cfg), 
        # because Dassl might build an evaluator in TrainerX.__init__
        self.lab2cname = {}
        
        self.cfg = cfg
        self.num_clients = cfg.FED.NUM_CLIENTS  # e.g. 2
        self.num_rounds = cfg.FED.NUM_ROUNDS
        self.local_epochs = cfg.FED.LOCAL_EPOCHS
        self.clients = []
        self.global_weights = None

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

        # -- 2) Load UcMerced
        uc_cfg = self.cfg.clone()
        uc_cfg.defrost()
        uc_cfg.DATASET.NAME = "Ucmerced"
        dm_uc = DataManager(uc_cfg)
        dataset_uc = dm_uc.dataset

        # local label->classname
        pn_lab2cname = dm_pn.lab2cname
        uc_lab2cname = dm_uc.lab2cname

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

        # -- 3) Form global list of classes (union)
        set_pn = set(pn_lab2cname.values())
        set_uc = set(uc_lab2cname.values())
        global_list = sorted(list(set_pn.union(set_uc)))
        global_num_classes = len(global_list)
        print(f"[INFO] Unified #classes = {global_num_classes}")

        # Build name->gid
        name2gid = {cname: i for i, cname in enumerate(global_list)}

        # Save a dictionary {class_id -> class_name} for Dassl
        self.lab2cname = {i: cname for i, cname in enumerate(global_list)}

        # -- 4) Remap local labels -> global IDs
        def remap(data_list, local_lab2cname):
            for idx, item in enumerate(data_list):
                old_label = item.label
                cname = local_lab2cname[old_label]
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

        self.client_data_managers = [dm_client_0, dm_client_1]

        # aggregator-level loaders not needed
        self.train_loader_x = None
        self.val_loader = None
        self.test_loader = None
        self.dm = None
        self.debug_clients_data()

    ###################################################
    # B) Build local trainers (MaPLe)
    ###################################################
    def build_model(self):
        self.clients = []
        # We'll keep the global classnames in a variable to pass along
        global_classnames = list(self.lab2cname.values())

        for i, dm in enumerate(self.client_data_managers):
            local_trainer = MaPLe(self.cfg, client_id=i, classnames=global_classnames)
            local_trainer.dm = dm
            local_trainer.build_model()
            self.clients.append(local_trainer)

        # initialize global weights from client 0
        self.global_weights = self.clients[0].model.state_dict()

    ###################################################
    # C) Federated training loop
    ###################################################
    """def train(self):
        for round_idx in range(self.num_rounds):
            print(f"\n--- Federated Round {round_idx+1}/{self.num_rounds} ---")

            # 1) broadcast
            if self.check_weights_valid(self.global_weights):
                self.broadcast_weights(self.global_weights)
            else:
                print("Invalid global weights detected!")
                self.nan_stats["skipped_rounds"] += 1
                continue

            # 2) local training
            local_state_dicts = []
            valid_clients = 0
            for i, trainer in enumerate(self.clients):
                print(f"[Client {i}] local training ...")
                trainer.epoch = round_idx * self.local_epochs
                trainer.max_epoch = (round_idx + 1) * self.local_epochs

                try:
                    for ep in range(trainer.epoch, trainer.max_epoch):
                        trainer.run_epoch(ep)
                except RuntimeError as e:
                    print(f"Client {i} failed training: {e}")
                    self.nan_stats["failed_clients"].append(i)
                    continue

                # gather weights
                w = trainer.model.state_dict()
                if self.check_weights_valid(w):
                    local_state_dicts.append(w)
                    valid_clients += 1
                else:
                    print(f"Client {i} produced invalid weights, skipping agg")
                    trainer.model.load_state_dict(self.global_weights)

            # 3) FedAvg if possible
            if valid_clients > 0:
                self.global_weights = self.safe_average_weights(local_state_dicts, valid_clients)
                self.nan_stats["total_updates"] += 1
            else:
                print("All clients failed! Revert to previous global.")
                self.nan_stats["skipped_rounds"] += 1

        # done
        self.finalize_training()"""
    def train(self):
        for round_idx in range(self.num_rounds):
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

                # Collect weights
                w = trainer.model.state_dict()
                if self.check_weights_valid(w):
                    local_state_dicts.append(w)
                    valid_clients += 1
                else:
                    print(f"Client {i} produced invalid weights, skipping aggregation")
                    trainer.model.load_state_dict(self.global_weights)

            # 3) Print average local training loss for this round
            if round_losses:
                avg_loss_this_round = sum(round_losses) / len(round_losses)
                print(f"[Round {round_idx+1}] Avg local training loss = {avg_loss_this_round:.4f}")

            # 4) Perform FedAvg if possible
            if valid_clients > 0:
                self.global_weights = self.safe_average_weights(local_state_dicts, valid_clients)
                self.nan_stats['total_updates'] += 1
            else:
                print("All clients failed! Reverting to previous global model.")
                self.nan_stats['skipped_rounds'] += 1

            # 5) (Optional) Evaluate test accuracy after each round using client 0
            if self.check_weights_valid(self.global_weights):
                self.broadcast_weights(self.global_weights)
                # Evaluate on client 0's test set (or loop all clients if desired)
                test_res = self.clients[0].test()
                if "accuracy" in test_res:
                    print(f"[Round {round_idx+1}] Test accuracy (client 0) = {test_res['accuracy']:.2f}%")
            else:
                print("Global weights invalid after aggregation, skipping test.")

        # 6) Done training
        self.finalize_training()


    ###################################################
    # D) Utility functions
    ###################################################
    def safe_average_weights(self, local_dicts, valid_clients):
        avg_state = {}
        for key in local_dicts[0].keys():
            stacked = torch.stack([sd[key].float() for sd in local_dicts])
            stacked = torch.nan_to_num(stacked, nan=0.0, posinf=1e4, neginf=-1e4)
            avg_state[key] = torch.mean(stacked, dim=0).half()
        return avg_state

    def check_weights_valid(self, state_dict):
        for name, param in state_dict.items():
            if torch.isnan(param).any():
                print(f"NaN in {name}")
                return False
            if torch.isinf(param).any():
                print(f"Inf in {name}")
                return False
        return True

    def broadcast_weights(self, global_sd):
        """Broadcast to each client with strict=True, ensuring same shape."""
        for client_trainer in self.clients:
            client_trainer.model.load_state_dict(global_sd, strict=True)
            # reset optimizer states
            for param_group in client_trainer.optim.param_groups:
                for p in param_group["params"]:
                    if p in client_trainer.optim.state:
                        del client_trainer.optim.state[p]
            # rebuild scheduler
            client_trainer.sched = build_lr_scheduler(client_trainer.optim, client_trainer.cfg.OPTIM)
            if hasattr(client_trainer, "epoch"):
                client_trainer.sched.last_epoch = client_trainer.epoch - 1

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
        save_checkpoint(checkpoint, target_dir, is_best=is_best)
        if self.cfg.VERBOSE:
            print(f"Model saved to {target_dir}")

    def load_model(self, directory, epoch=None):
        """Load aggregator weights from disk."""
        if not directory:
            print("Skipping load_model, no pretrained path given")
            return
        subfolder = "MultiModalPromptLearner_Aggregator"
        if epoch is not None:
            model_file = f"model.pth.tar-{epoch}"
        else:
            model_file = "model.pth.tar"
        path = osp.join(directory, subfolder, model_file)
        if not osp.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")
        ckpt = load_checkpoint(path)
        state_dict = ckpt["state_dict"]
        loaded_epoch = ckpt.get("epoch", None)

        self.global_weights = state_dict
        print(f"Loaded aggregator weights from '{path}' (epoch={loaded_epoch}).")
        if self.check_weights_valid(self.global_weights):
            self.broadcast_weights(self.global_weights)
            print("Broadcasted loaded global weights.")
        else:
            print("Warning: loaded global weights invalid! Skipping broadcast.")


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
        """Call debug_print_samples on each client's data manager."""
        for i, dm in enumerate(self.client_data_managers):
            print(f"\n=== Client {i} ===")
            self.debug_save_samples_images(dm, subset="train_x", max_per_class=5)
            self.debug_save_samples_images(dm, subset="val",     max_per_class=5)
            self.debug_save_samples_images(dm, subset="test",    max_per_class=5)

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


