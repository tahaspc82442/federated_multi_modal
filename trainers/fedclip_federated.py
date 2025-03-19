# # # trainers/fedclip_federated.py

# # from dassl.engine import TRAINER_REGISTRY, TrainerX
# # from dassl.data import DataManager
# # import copy
# # import torch
# # import torch.nn.functional as F
# # from trainers.fedclip import FedCLIP  # Import the corrected FedCLIP
# # import clip
# # #import wandb
# # from dassl.data.datasets import build_dataset
# # from dassl.data.data_manager import DatasetWrapper
# # from torch.utils.data import DataLoader

# # @TRAINER_REGISTRY.register()
# # class FedCLIPFederated(TrainerX):
# #     def __init__(self, cfg):
# #         super().__init__(cfg)
# #         self.clients = []
# #         self.global_model = None
# #         self.num_clients = cfg.FED.NUM_CLIENTS
# #         self.local_epochs = cfg.FED.LOCAL_EPOCHS
# #         self.num_rounds = cfg.FED.NUM_ROUNDS
# #         self.device = cfg.MODEL.DEVICE #get device

# #     def build_data_loader(self):
# #         """Builds client-specific data loaders."""
# #         cfg = self.cfg
        
# #         #1. Assign datasets to clients.  REPLACE THESE with your actual dataset names.
# #         client_datasets = [
# #             "PatternNet", "Ucmerced", "EuroSAT", "Mlrs", "Milaid" 
# #         ]


        
# #         if len(client_datasets) != cfg.FED.NUM_CLIENTS:
# #             raise ValueError("Number of client datasets must match the number of clients.")
        
# #         root = cfg.DATASET.ROOT
        
# #         self.client_data_managers = [] # Initialize the list to store datamanagers

# #         for i, dataset_name in enumerate(client_datasets):
# #             # 2. Load dataset config for the specific client
# #             cfg.defrost()
# #             cfg.DATASET.NAME = dataset_name #Change dataset name dynamically
# #             cfg.freeze()
            
# #             # Build the dataset
# #             dataset = build_dataset(cfg)

# #             # Build train, val, and test sets
# #             train_x = dataset.train_x
# #             val = dataset.val
# #             test = dataset.test
            
# #             # Build the data manager
# #             data_manager = DataManager(cfg, dataset)
            
# #             # Get train, val, and test loaders from the data manager
# #             train_loader_x = data_manager.train_loader_x
# #             val_loader = data_manager.val_loader
# #             test_loader = data_manager.test_loader

# #             #Append to the datamanager
# #             self.client_data_managers.append(data_manager)

# #             print(f"Client {i+1}: Dataset: {dataset_name}, Train size: {len(train_x)}, Val size: {len(val)}, Test size: {len(test)}")
        
# #         #Creating the test data loader, concatinating test data from all clients.
# #         test_data = []
# #         for dm in self.client_data_managers:
# #             test_data.extend(dm.dataset.test)

# #         random_seed = cfg.SEED if cfg.SEED >= 0 else 42
# #         test_data = list(test_data)
# #         test_dataset = DatasetWrapper(cfg, test_data, test_data, transform=dataset.transform_test,
# #                                             is_train=False, return_idx=False)
# #         self.test_loader = DataLoader(
# #                 test_dataset,
# #                 batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
# #                 sampler=data_manager.test_sampler,
# #                 num_workers=cfg.DATALOADER.NUM_WORKERS,
# #                 drop_last=False,
# #                 pin_memory=True,
# #             )

# #     def build_model(self):
# #         # Initialize global model
# #         cfg = self.cfg
# #         self.global_model = FedCLIP(cfg.MODEL.BACKBONE.NAME)
# #         self.global_model.to(self.device) #Move model to device
# #         self.clients = [copy.deepcopy(self.global_model) for _ in range(self.num_clients)]

# #     def train(self):
# #         cfg = self.cfg #For ease of use
# #         for round_idx in range(self.num_rounds):
# #             print(f"Round {round_idx + 1}/{self.num_rounds}")
# #             # Broadcast global adapter weights
# #             self._broadcast_weights()

# #             # Client training
# #             client_updates = []
# #             for client_idx in range(self.num_clients):
# #                 client_model = self.clients[client_idx]
# #                 client_model.to(self.device) #Move to device
# #                 client_loader = self.client_data_managers[client_idx].train_loader_x #Corrected loader
# #                 client_updates.append(self._train_client(client_model, client_loader))

# #             # Aggregate updates
# #             self._aggregate(client_updates)

# #             # Evaluate
# #             if (round_idx + 1) % cfg.FED.EVAL_FREQ == 0:
# #                 test_acc = self.evaluate()
# #                 # Log to WandB
# #                 #wandb.log({"round": round_idx + 1, "test_accuracy": test_acc})

# #     def _train_client(self, model, loader):
# #         model.train()
# #         optimizer = torch.optim.Adam(model.img_adap.parameters(), lr=self.cfg.OPTIM.LR)

# #         for epoch in range(self.local_epochs): #Iterate over epochs
# #           for batch in loader:
# #               images = batch['img'].to(self.device) #Correct key
# #               texts = clip.tokenize(batch['label_text']).to(self.device)   # Tokenize here

# #               optimizer.zero_grad()
# #               logits = model(images, texts)
# #               # Create labels for cross-entropy.  Assumes batch size is the same as the number of classes *for that client*
# #               labels = torch.arange(len(images)).to(self.device)
# #               loss = F.cross_entropy(logits, labels)
# #               loss.backward()
# #               optimizer.step()
# #         return model.img_adap.state_dict()


# #     def _broadcast_weights(self):
# #         global_weights = self.global_model.img_adap.state_dict()
# #         for client in self.clients:
# #             client.img_adap.load_state_dict(global_weights)

# #     def _aggregate(self, updates):
# #         avg_weights = {}
# #         for key in updates[0].keys():
# #             avg_weights[key] = torch.mean(
# #                 torch.stack([update[key].to(self.device) for update in updates]), dim=0 #Added device
# #             )
# #         self.global_model.img_adap.load_state_dict(avg_weights)

# #     def evaluate(self):
# #         self.global_model.eval()
# #         correct = 0
# #         total = 0

# #         with torch.no_grad():
# #             for batch in self.test_loader:
# #                 images = batch['img'].to(self.device)
# #                 texts = clip.tokenize(batch['label_text']).to(self.device) #Tokenize here
# #                 labels = batch['label'].to(self.device) #Should be numeric labels


# #                 logits = self.global_model(images, texts)
# #                 preds = logits.argmax(dim=1)
# #                 correct += (preds == labels).sum().item()
# #                 total += labels.size(0)

# #         acc = 100 * correct / total
# #         print(f"Test Accuracy: {acc:.2f}%")
# #         return acc


# # trainers/fedclip_federated.py

# from dassl.engine import TRAINER_REGISTRY, TrainerX
# from dassl.data import DataManager
# import copy
# import torch
# import torch.nn.functional as F
# from trainers.fedclip import FedCLIP
# import clip
# #import wandb #Commented Wandb
# from dassl.data.datasets import build_dataset
# from dassl.data.data_manager import DatasetWrapper, build_transform  # Import build_transform
# from torch.utils.data import DataLoader


# class FedCLIPDatasetWrapper(DatasetWrapper):  # Create a custom wrapper
#     def __init__(self, cfg, dataset, transform=None, is_train=False, return_idx=False):
#         super().__init__(cfg, dataset.train_x, dataset.val, dataset.test,
#                          transform=transform, is_train=is_train,
#                          return_idx=return_idx)
#         self.dataset = dataset
#         self.transform_train = transform[0] if transform else None #Get individual transforms
#         self.transform_test = transform[1] if transform else None

#     def __getitem__(self, index):
#         #Determine if train or test
#         if self.is_train:
#             item = self.dataset.train_x[index]
#             img = Image.open(item.impath).convert("RGB")
#             if self.transform_train:
#                 img = self.transform_train(img)
#             output = {
#                 'img': img,
#                 'label': item.label,
#                 'label_text': item.classname, # Use class name as the text
#                 'impath': item.impath
#                 }

#         else:
#             if index < len(self.dataset.test): #For test set
#                 item = self.dataset.test[index]
#                 img = Image.open(item.impath).convert("RGB")
#                 if self.transform_test:
#                     img = self.transform_test(img)
#                 output = {
#                     'img': img,
#                     'label': item.label,
#                     'label_text': item.classname,
#                     'impath': item.impath
#                     }
#             else: #For val set
#                 item = self.dataset.val[index - len(self.dataset.test)]
#                 img = Image.open(item.impath).convert("RGB")
#                 if self.transform_test:
#                     img = self.transform_test(img)
#                 output = {
#                    'img': img,
#                     'label': item.label,
#                     'label_text': item.classname,
#                     'impath': item.impath
#                 }

#         return output


# @TRAINER_REGISTRY.register()
# class FedCLIPFederated(TrainerX):
#     def __init__(self, cfg):
#         super().__init__(cfg)
#         self.clients = []
#         self.global_model = None
#         self.num_clients = cfg.FED.NUM_CLIENTS
#         self.local_epochs = cfg.FED.LOCAL_EPOCHS
#         self.num_rounds = cfg.FED.NUM_ROUNDS
#         self.device = cfg.MODEL.DEVICE

#     def build_data_loader(self):
#         """Builds client-specific data loaders."""
#         cfg = self.cfg

#         client_datasets = [
#             "PatternNet", "Ucmerced", "EuroSAT", "Mlrs", "Milaid"
#         ]

#         if len(client_datasets) != cfg.FED.NUM_CLIENTS:
#             raise ValueError("Number of client datasets must match the number of clients.")

#         root = cfg.DATASET.ROOT
#         self.client_data_managers = []

#         for i, dataset_name in enumerate(client_datasets):
#             cfg.defrost()
#             cfg.DATASET.NAME = dataset_name
#             cfg.freeze()
#             dataset = build_dataset(cfg)

#             # Build transformations *here*, using DASSL's build_transform
#             transform_train = build_transform(cfg, is_train=True)
#             transform_test = build_transform(cfg, is_train=False)

#              # Use the custom wrapper
#             data_manager = DataManager(cfg, dataset, custom_tfm_train=transform_train, custom_tfm_test=transform_test)
#             train_loader_x = data_manager.train_loader_x
#             val_loader = data_manager.val_loader
#             test_loader = data_manager.test_loader
#             self.client_data_managers.append(data_manager)

#             print(f"Client {i+1}: Dataset: {dataset_name}, Train size: {len(dataset.train_x)}, Val size: {len(dataset.val)}, Test size: {len(dataset.test)}")

#         test_data = []
#         for dm in self.client_data_managers:
#             test_data.extend(dm.dataset.test)


#         # Build the combined test loader using the custom wrapper, passing
#         transform_test = build_transform(cfg, is_train=False)
#         random_seed = cfg.SEED if cfg.SEED >= 0 else 42
#         test_data = list(test_data)
#         test_dataset = FedCLIPDatasetWrapper(cfg, dataset, transform=[None, transform_test], is_train=False)
#         self.test_loader = DataLoader(
#                 test_dataset,
#                 batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
#                 #sampler=data_manager.test_sampler, #Cannot use test sampler here
#                 num_workers=cfg.DATALOADER.NUM_WORKERS,
#                 drop_last=False,
#                 pin_memory=True,
#             )

#     def build_model(self):
#         cfg = self.cfg
#         self.global_model = FedCLIP(cfg.MODEL.BACKBONE.NAME)
#         self.global_model.to(self.device)
#         self.clients = [copy.deepcopy(self.global_model) for _ in range(self.num_clients)]

#     def train(self):
#         cfg = self.cfg
#         for round_idx in range(self.num_rounds):
#             print(f"Round {round_idx + 1}/{self.num_rounds}")
#             self._broadcast_weights()
#             client_updates = []
#             for client_idx in range(self.num_clients):
#                 client_model = self.clients[client_idx]
#                 client_model.to(self.device)
#                 client_loader = self.client_data_managers[client_idx].train_loader_x
#                 client_updates.append(self._train_client(client_model, client_loader))
#             self._aggregate(client_updates)
#             if (round_idx + 1) % cfg.FED.EVAL_FREQ == 0:
#                 test_acc = self.evaluate()
#                 #wandb.log({"round": round_idx + 1, "test_accuracy": test_acc}) #Commented wandb

#     def _train_client(self, model, loader):
#         model.train()
#         optimizer = torch.optim.Adam(model.img_adap.parameters(), lr=self.cfg.OPTIM.LR)
#         for epoch in range(self.local_epochs):
#           for batch in loader:
#               images = batch['img'].to(self.device)
#               texts = clip.tokenize(batch['label_text']).to(self.device)
#               optimizer.zero_grad()
#               logits = model(images, texts)
#               labels = torch.arange(len(images)).to(self.device)
#               loss = F.cross_entropy(logits, labels)
#               loss.backward()
#               optimizer.step()
#         return model.img_adap.state_dict()

#     def _broadcast_weights(self):
#         global_weights = self.global_model.img_adap.state_dict()
#         for client in self.clients:
#             client.img_adap.load_state_dict(global_weights)

#     def _aggregate(self, updates):
#         avg_weights = {}
#         for key in updates[0].keys():
#             avg_weights[key] = torch.mean(
#                 torch.stack([update[key].to(self.device) for update in updates]), dim=0
#             )
#         self.global_model.img_adap.load_state_dict(avg_weights)

#     def evaluate(self):
#         self.global_model.eval()
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for batch in self.test_loader:
#                 images = batch['img'].to(self.device)
#                 texts = clip.tokenize(batch['label_text']).to(self.device)
#                 labels = batch['label'].to(self.device)
#                 logits = self.global_model(images, texts)
#                 preds = logits.argmax(dim=1)
#                 correct += (preds == labels).sum().item()
#                 total += labels.size(0)
#         acc = 100 * correct / total
#         print(f"Test Accuracy: {acc:.2f}%")
#         return acc


# trainers/fedclip_federated.py


# trainers/fedclip_federated.py

# trainers/fedclip_federated.py
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.data import DataManager
from dassl.data.datasets import Datum
from dassl.utils import (
    mkdir_if_missing, 
    load_checkpoint,
    save_checkpoint
)
import copy
import torch
import torch.nn.functional as F
from trainers.fedclip import FedCLIP
import clip
#import wandb
from dassl.data.datasets import build_dataset
from dassl.data.data_manager import DatasetWrapper, build_data_loader
from torch.utils.data import DataLoader
from PIL import Image
import os
from dassl.data.transforms import build_transform


# trainers/fedclip_federated.py




@TRAINER_REGISTRY.register()
class FedCLIPFederated(TrainerX):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.clients = []
        self.global_model = None
        # These are now correctly initialized *after* the super().__init__ call
        self.num_clients = cfg.FED.NUM_CLIENTS  # Correct
        self.local_epochs = cfg.FED.LOCAL_EPOCHS
        self.num_rounds = cfg.FED.NUM_ROUNDS
        self.device = cfg.MODEL.DEVICE
        self.build_model()  # Call build_model here
        print(f"After build_model: self.global_model = {self.global_model}")

    def build_data_loader(self):
        """Builds client-specific data loaders."""
        cfg = self.cfg  # Access the config object

        client_datasets = [
            "PatternNet", "Ucmerced", "EuroSAT", "Mlrs", "Milaid"
        ]

        # Access num_clients directly from cfg *within* build_data_loader
        if len(client_datasets) != cfg.FED.NUM_CLIENTS:  # Correct: use cfg
            raise ValueError("Number of client datasets must match the number of clients.")

        root = cfg.DATASET.ROOT
        self.client_data_managers = []

        # --- Class Merging Logic ---
        all_classnames = set()

        for i, dataset_name in enumerate(client_datasets):
            cfg.defrost()
            cfg.DATASET.NAME = dataset_name
            cfg.freeze()
            dataset = build_dataset(cfg)
            all_classnames.update(dataset.classnames)

            # Build transformations
            transform_train = build_transform(cfg, is_train=True)
            transform_test = build_transform(cfg, is_train=False)

            # Create DataManager and loaders using build_data_loader
            train_loader_x = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TRAIN_X.SAMPLER,
                data_source=dataset.train_x,
                batch_size=cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
                tfm=transform_train,
                is_train=True,
            )
            val_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.val,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=transform_test,
                is_train=False,
            )
            test_loader = build_data_loader(
                cfg,
                sampler_type=cfg.DATALOADER.TEST.SAMPLER,
                data_source=dataset.test,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=transform_test,
                is_train=False,
             )
            # Create a dictionary to store loaders
            data_manager = {
                "train_loader_x": train_loader_x,
                "val_loader": val_loader,
                "test_loader": test_loader,
                "dataset": dataset
            }


            self.client_data_managers.append(data_manager)
            print(f"Client {i+1}: Dataset: {dataset_name}, Train size: {len(dataset.train_x)}, Val size: {len(dataset.val)}, Test size: {len(dataset.test)}")

        # --- Global Label Mapping ---
        global_class_list = sorted(list(all_classnames))
        name2globalid = {cname: i for i, cname in enumerate(global_class_list)}
        self.lab2cname = {i: cname for i, cname in enumerate(global_class_list)}  # Needed for TrainerX
        cfg.defrost()
        cfg.MODEL.NUM_CLASSES = len(global_class_list)
        cfg.freeze()

        # Create the combined test loader
        test_data = []
        for dm in self.client_data_managers:
             test_data.extend(dm["dataset"].test)

        # Apply global label remapping, create new Datum objects
        remapped_test_data = []
        for item in test_data:
            if item.classname in name2globalid:
                new_label = name2globalid[item.classname]
                new_item = Datum(impath=item.impath, label=new_label, classname=item.classname, caption=item.caption)
                remapped_test_data.append(new_item)
            else:
                remapped_test_data.append(item)

        test_dataset = DatasetWrapper(cfg, remapped_test_data, transform=transform_test, is_train=False)
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            drop_last=False,
            pin_memory=True,
        )

    def build_model(self):
        cfg = self.cfg
        print("Inside build model")
        self.global_model = FedCLIP(cfg.MODEL.BACKBONE.NAME)
        print(f"After FedCLIP init: self.global_model = {self.global_model}")
        self.global_model.to(self.device)
        self.clients = [copy.deepcopy(self.global_model) for _ in range(cfg.FED.NUM_CLIENTS)]   


    def train(self):
        cfg = self.cfg
        for round_idx in range(self.num_rounds):
            print(f"Round {round_idx + 1}/{self.num_rounds}")
            self._broadcast_weights()
            client_updates = []
            for client_idx in range(cfg.FED.NUM_CLIENTS):
                client_model = self.clients[client_idx]
                client_model.to(self.device)
                client_loader = self.client_data_managers[client_idx]["train_loader_x"]
                client_updates.append(self._train_client(client_model, client_loader, self.client_data_managers[client_idx]["dataset"].classnames))
            self._aggregate(client_updates)
            if (round_idx + 1) % cfg.FED.EVAL_FREQ == 0:
                test_acc = self.evaluate(self.test_loader, self.lab2cname.values())
                #wandb.log({"round": round_idx + 1, "test_accuracy": test_acc})

    def _train_client(self, model, loader, classnames):
        model.train()
        optimizer = torch.optim.Adam(model.img_adap.parameters(), lr=self.cfg.OPTIM.LR, weight_decay=self.cfg.OPTIM.WEIGHT_DECAY)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classnames]).to(self.device)
        for _ in range(self.local_epochs):
          for batch in loader:
              images = batch['img'].to(self.device)
              optimizer.zero_grad()
              logits = model(images, text_inputs)
              labels = batch['label'].to(self.device)
              loss = F.cross_entropy(logits, labels)
              loss.backward()
              optimizer.step()
        return model.img_adap.state_dict()

    def _broadcast_weights(self):
        global_weights = self.global_model.img_adap.state_dict()
        for client in self.clients:
            client.img_adap.load_state_dict(global_weights)

    def _aggregate(self, updates):
        avg_weights = {}
        for key in updates[0].keys():
            avg_weights[key] = torch.mean(
                torch.stack([update[key].to(self.device) for update in updates]), dim=0
            )
        self.global_model.img_adap.load_state_dict(avg_weights)

    def evaluate(self, test_loader, classnames):

        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in classnames]).to(self.device)
        self.global_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                images = batch['img'].to(self.device)
                labels = batch['label'].to(self.device)
                logits = self.global_model(images, text_inputs)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        acc = 100 * correct / total
        print(f"Test Accuracy: {acc:.2f}%")
        return acc