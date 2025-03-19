# # Add these imports at the top
# import clip
# from fedclip.nets.models import ClipModelat
# from fedclip.utils.clip_util import get_text_features_list, get_similarity
# import torch
# import wandb
# import os.path as osp
# import torch
# import numpy as np
# import torch.nn.functional as F

# from dassl.engine import TRAINER_REGISTRY, TrainerX
# from tqdm import trange
# import copy


# @TRAINER_REGISTRY.register()
# class FedCLIPFederated(TrainerX):
#     def __init__(self, cfg):
#         super().__init__(cfg)
#         # Add CLIP-specific initialization
#         self.clip_model = None
#         self.text_features = None
#         self.class_prompts = ["a satellite image of {}".format(cname) for cname in self.lab2cname.values()]  # Customize for remote sensing

#     def build_data_loader(self):
#         # Keep your existing data merging logic but add CLIP preprocessing
#         super().build_data_loader()
        
#         # Add CLIP-specific transforms
#         self.clip_preprocess = clip.load("ViT-B/32")[1]
        
#         # Modify datasets to include CLIP-compatible prompts
#         for dm in self.client_data_managers:
#             dm.dataset.train_x = self._add_clip_prompts(dm.dataset.train_x)
#             dm.dataset.val = self._add_clip_prompts(dm.dataset.val)
#             dm.dataset.test = self._add_clip_prompts(dm.dataset.test)

#     def _add_clip_prompts(self, data_list):
#         new_data = []
#         for datum in data_list:
#             # Create CLIP-compatible prompt
#             text = clip.tokenize([f"a satellite image of {datum.classname}"])
#             new_data.append((datum.impath, text, datum.label))
#         return new_data

#     def build_model(self):
#         # Initialize global CLIP model with adapters
#         self.clip_model = ClipModelat(
#             self.cfg.MODEL.BACKBONE.NAME,  # e.g., "ViT-B/32"
#             device=self.cfg.device,
#             imgadpy=True,
#             freezepy=True
#         )
        
#         # Initialize text features
#         with torch.no_grad():
#             self.text_features = get_text_features_list(
#                 self.class_prompts,
#                 self.clip_model.model
#             ).to(self.cfg.device)

#         # Create client-specific models
#         self.clients = [copy.deepcopy(self.clip_model) for _ in self.client_data_managers]

#     def train(self):
#         for round_idx in trange(self.num_rounds):
#             print(f"\n--- Federated Round {round_idx+1}/{self.num_rounds} ---")

#             # 1) Broadcast global adapter weights
#             self._broadcast_adapters()

#             # 2) Client training
#             client_weights = []
#             for i, client_model in enumerate(self.clients):
#                 print(f"Training client {i}")
#                 self._train_client(client_model, self.client_data_managers[i])
#                 client_weights.append(client_model.img_adap.state_dict())

#             # 3) Aggregate adapters
#             self._aggregate_adapters(client_weights)

#             # 4) Evaluate
#             if round_idx % 5 == 0:
#                 self._evaluate_round(round_idx)

#     def _train_client(self, model, data_manager):
#         model.train()
#         optimizer = torch.optim.Adam(
#             model.img_adap.parameters(),
#             lr=self.cfg.OPTIM.LR,
#             weight_decay=self.cfg.OPTIM.WEIGHT_DECAY
#         )

#         for epoch in range(self.local_epochs):
#             for batch in data_manager.train_loader:
#                 images = batch['images'].to(self.cfg.device)
#                 texts = batch['texts'].to(self.cfg.device)
#                 labels = batch['labels'].to(self.cfg.device)

#                 # Forward pass
#                 image_features = model.model.encode_image(images)
#                 text_features = model.model.encode_text(texts)
                
#                 # Adapter processing
#                 image_features_att = model.img_adap(image_features)
#                 adapted_features = image_features * image_features_att

#                 # Calculate loss
#                 logit_scale = model.model.logit_scale.exp()
#                 logits_per_image = logit_scale * adapted_features @ text_features.t()
#                 logits_per_text = logits_per_image.t()
                
#                 loss = (F.cross_entropy(logits_per_image, labels) +
#                        F.cross_entropy(logits_per_text, labels)) / 2

#                 # Backward pass
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#     def _broadcast_adapters(self):
#         """Share adapter weights with all clients"""
#         global_weights = self.clip_model.img_adap.state_dict()
#         for client in self.clients:
#             client.img_adap.load_state_dict(global_weights)

#     def _aggregate_adapters(self, client_weights):
#         """FedAvg on adapter weights"""
#         avg_weights = {}
#         for key in client_weights[0].keys():
#             avg_weights[key] = torch.mean(
#                 torch.stack([w[key] for w in client_weights]), dim=0
#             )
#         self.clip_model.img_adap.load_state_dict(avg_weights)

#     def _evaluate_round(self, round_idx):
#         self.clip_model.eval()
#         correct = 0
#         total = 0
        
#         with torch.no_grad():
#             for batch in self.client_data_managers[0].test_loader:
#                 images = batch['images'].to(self.cfg.device)
#                 labels = batch['labels'].to(self.cfg.device)
                
#                 # Forward pass
#                 image_features = self.clip_model.model.encode_image(images)
#                 image_features_att = self.clip_model.img_adap(image_features)
#                 adapted_features = image_features * image_features_att
                
#                 # Calculate similarity
#                 similarity = (adapted_features @ self.text_features.T).softmax(dim=-1)
#                 preds = similarity.argmax(dim=1)
                
#                 correct += (preds == labels).sum().item()
#                 total += labels.size(0)
                
#         acc = 100 * correct / total
#         print(f"[Round {round_idx+1}] Test Accuracy: {acc:.2f}%")
#         wandb.log({"round": round_idx, "test_accuracy": acc})

#     # Keep existing utility methods (save_model, load_model, etc.) but ensure they
#     # handle the CLIP adapter weights appropriately



import clip
import torch
import torch.nn as nn

# trainers/fedclip.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from clip import model  # Import the model module from CLIP

class FedCLIP(nn.Module):
    def __init__(self, clip_model_name):
        super().__init__()
        self.clip, self.preprocess = clip.load(clip_model_name) # Pass design_details
        # self.img_adap = nn.Linear(768, 768)  # Example – adjust as needed
        if clip_model_name == "ViT-B/32":
            self.img_adap = nn.Linear(512, 512)
        elif clip_model_name == "ViT-B/16":
             self.img_adap = nn.Linear(512, 512)
        elif clip_model_name == "ViT-L/14":
              self.img_adap = nn.Linear(768,768)
        else:
             raise ValueError("Invalid clip_model_name")
        self.logit_scale = nn.Parameter(torch.ones([]) * 4.60517)  # Example

    def forward(self, image, text):
        with torch.no_grad():  # Disable gradients for CLIP backbone
            image_features = self.clip.encode_image(image)
            text_features = self.clip.encode_text(text)

        image_features = image_features.float()
        text_features = text_features.float()

        image_features = self.img_adap(image_features)  # Example adaptation layer

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image