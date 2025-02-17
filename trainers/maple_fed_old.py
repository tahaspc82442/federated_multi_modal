# trainers/maple_fed.py

from dassl.engine import TRAINER_REGISTRY, TrainerX
from trainers.maple import MaPLe
from dassl.data import DataManager
from .data_partition import partition_dataset_iid
from .client_datamanager import ClientDataManager
import torch

from dassl.optim import build_optimizer, build_lr_scheduler

@TRAINER_REGISTRY.register()
class MaPLeFederated(TrainerX):
    """Federated trainer that orchestrates data partition + FedAvg for MaPLe."""

    def __init__(self, cfg):
        """
        1) We must define these attributes BEFORE calling super().__init__(cfg),
           because the parent TrainerX.__init__() will call build_data_loader(), 
           build_model(), etc.
        """
        self.cfg = cfg
        self.num_clients = cfg.FED.NUM_CLIENTS
        self.num_rounds = cfg.FED.NUM_ROUNDS
        self.local_epochs = cfg.FED.LOCAL_EPOCHS

        # We'll keep a list of local trainers and the global weights
        self.clients = []
        self.global_weights = None

        # Then call the parent's constructor (which triggers build_data_loader() -> build_model())
        super().__init__(cfg)

    def build_data_loader(self):
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

    """def train(self):
        
       # 4) The main federated loop:
         #  - For each round:
            # a) broadcast global weights
           #  b) local training
           #  c) FedAvg
           #- Finally, test the final global model on client 0
        
        for round_idx in range(self.num_rounds):
            print(f"\n--- Federated Round {round_idx+1}/{self.num_rounds} ---")

            # 4.1) Broadcast global weights to each client
            self.broadcast_weights(self.global_weights)

            # 4.2) Each client trains locally
            local_state_dicts = []
            for i, client_trainer in enumerate(self.clients):
                print(f"[Client {i}] Local training ...")

                # Optionally set local epoch bounds
                client_trainer.epoch = round_idx * self.local_epochs
                client_trainer.max_epoch = (round_idx+1) * self.local_epochs

                # Run local epochs
                client_trainer.run_epoch()  # single or multiple epochs, depending on your config

                # Collect updated local weights
                local_weights = client_trainer.model.state_dict()
                local_state_dicts.append(local_weights)

            # 4.3) FedAvg step
            self.global_weights = self.average_weights(local_state_dicts)

        # 4.4) Test final global model on client 0
        self.broadcast_weights(self.global_weights)
        self.clients[0].test()"""
    

    def train(self):
        """
        Main federated training loop:
        - For each round:
        a) Broadcast global weights
        b) Local training
        c) FedAvg
        - Finally, test the final global model on client 0.
        """
        for round_idx in range(self.num_rounds):
            print(f"\n--- Federated Round {round_idx+1}/{self.num_rounds} ---")
            self.check_model_weights(self.global_weights, tag="global before broadcast")
            # 1. Broadcast global weights to each client
            self.broadcast_weights(self.global_weights)

            # 2. Each client trains locally
            local_state_dicts = []
            for i, client_trainer in enumerate(self.clients):
                print(f"[Client {i}] Local training ...")

                # Update epoch bounds for the current round
                client_trainer.epoch = round_idx * self.local_epochs
                client_trainer.max_epoch = (round_idx + 1) * self.local_epochs

                # Run local training for the defined epoch range
                for local_epoch in range(client_trainer.epoch, client_trainer.max_epoch):
                    self.check_model_weights(client_trainer.model.state_dict(), tag=f"client {i} before local training")
                    client_trainer.run_epoch(local_epoch)
                    self.check_model_weights(client_trainer.model.state_dict(), tag=f"client {i} after local training")

                # Collect updated local weights
                local_weights = client_trainer.model.state_dict()
                local_state_dicts.append(local_weights)

            # 3. FedAvg: Average the local weights to update the global model
            self.global_weights = self.average_weights(local_state_dicts)
            self.check_model_weights(self.global_weights, tag="global after aggregation")

        # 4. Test the final global model on client 0
        self.broadcast_weights(self.global_weights)
        self.clients[0].test()


    

    def average_weights(self, list_of_state_dicts):
        """Standard FedAvg: take a simple average of the local parameter dictionaries."""
        import copy
        avg_state = copy.deepcopy(list_of_state_dicts[0])
        for key in avg_state.keys():
            for i in range(1, len(list_of_state_dicts)):
                avg_state[key] += list_of_state_dicts[i][key]
            avg_state[key] = avg_state[key] / len(list_of_state_dicts)
        return avg_state

    def broadcast_weights(self, global_state_dict):
        """
        5) Overwrite each client's model with the global averaged parameters,
           then re-init the optimizer to keep momentum buffers in sync.
        """
        for client_trainer in self.clients:
            # Load the new global weights
            client_trainer.model.load_state_dict(global_state_dict, strict=False)

            # Re-init optimizer & scheduler so momentum states match the new params
            #client_trainer.optim = build_optimizer(client_trainer.model, client_trainer.cfg.OPTIM)
            #client_trainer.sched = build_lr_scheduler(client_trainer.optim, client_trainer.cfg.OPTIM)

            for group in client_trainer.optim.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        state = client_trainer.optim.state[p]
                        if 'momentum_buffer' in state:
                            state['momentum_buffer'] = p.data.clone()


    def check_model_weights(self,state_dict, tag=""):
        for name, param in state_dict.items():
            if torch.isnan(param).any():
                print(f"[DEBUG] NaN detected in weights: {name} {tag}")
            if torch.isinf(param).any():
                print(f"[DEBUG] Inf detected in weights: {name} {tag}")

