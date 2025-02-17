import os.path as osp
from collections import OrderedDict
import math
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
import numpy as np
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {"trainer": 'MaPLe',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.MAPLE.N_CTX}
    model = clip.build_model(state_dict or model.state_dict(), design_details)

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts, compound_prompts_deeper_text):
        # Step 1
        x = prompts + self.positional_embedding.type(self.dtype)
       # print(f"Step 1 (after positional embedding): size={x.size()}, dtype={x.dtype}")

        # Step 2
        x = x.permute(1, 0, 2)  # NLD -> LND
      #  print(f"Step 2 (after permute NLD -> LND): size={x.size()}, dtype={x.dtype}")

        # Step 3
        combined = [x, compound_prompts_deeper_text, 0]
        outputs = self.transformer(combined)
        x = outputs[0]  # extract the x back from here
     #   print(f"Step 3 (after transformer): size={x.size()}, dtype={x.dtype}")

        # Step 4
        x = x.permute(1, 0, 2)  # LND -> NLD
      #  print(f"Step 4 (after permute LND -> NLD): size={x.size()}, dtype={x.dtype}")

        # Step 5
        x = self.ln_final(x).type(self.dtype)
     #   print(f"Step 5 (after layer normalization): size={x.size()}, dtype={x.dtype}")

        # Step 6
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
      #  print(f"Step 6 (after text projection): size={x.size()}, dtype={x.dtype}")

        return x

class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MAPLE.N_CTX
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")

        self.proj_lang_to_vis = nn.Linear(ctx_dim, 768)
        self.proj_lang_to_vis.half()
        self.proj_vis_to_lang = nn.Linear(768, ctx_dim)
        self.proj_vis_to_lang.half()
        self.ctx = nn.Parameter(ctx_vectors)

        self.compound_prompts_text_parameters = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
                                                      for i in range(self.compound_prompts_depth - 1) if i%2==0])
        self.visual_deep_prompts_parameters = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 768)) for i in range(self.compound_prompts_depth - 1) if i%2!=0])
        
        for single_para in self.compound_prompts_text_parameters:
            nn.init.normal_(single_para, std=0.02)
        for single_para in self.visual_deep_prompts_parameters:
            nn.init.normal_(single_para, std=0.02)


        #print("printing CTX DIM", ctx_dim)

        self.compound_prompt_projections = nn.ModuleList([nn.Linear(ctx_dim, 768) if i % 2 == 0 
                                                          else nn.Linear(768, ctx_dim) 
                                                          for i in range(self.compound_prompts_depth-1)])
        #self.compound_prompts_text=[0 for i in range(len(self.compound_prompt_projections)-1)]
        self.compound_prompts_text=[0 for i in range(self.compound_prompts_depth-1)]


        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def print_info(self, var, name):
        if isinstance(var, list):
            return
            for i, v in enumerate(var):
                self.print_info(v, f"{name}[{i}]")
        else:
            return
            print(f"{name}: shape={var.shape if isinstance(var, torch.Tensor) else 'N/A'}, dtype={var.dtype if isinstance(var, torch.Tensor) else type(var)}")

    def forward(self):
        ctx = self.ctx
       # self.print_info(ctx, "Initial ctx")

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
            #self.print_info(ctx, "ctx after unsqueeze and expand")

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)
      #  self.print_info(prompts, "Prompts")

        visual_deep_prompts = [0 for i in range(self.compound_prompts_depth-1)] #[0 for i in range(len(self.compound_prompt_projections)-1)]
      #  print("length of compound prompts", len(self.compound_prompt_projections))
      #  print("length of visual_deep_prompts", len(visual_deep_prompts))
      #  print("length of compound_prompts_text", len(self.compound_prompts_text))
        for index, layer in enumerate(self.compound_prompt_projections):
            if index % 2 == 0:
                #print("shape", self.compound_prompts_text_parameters[index/2].shape)
                visual_prompt = layer(self.compound_prompts_text_parameters[int(index/2)])
                visual_deep_prompts[index] = visual_prompt
                self.compound_prompts_text[index] = self.compound_prompts_text_parameters[int(index/2)]
             #   self.print_info(visual_prompt, f"visual_deep_prompts[{index}]")
            else:
                
                #print("shape", self.visual_deep_prompts_parameters[index].shape)
                text_prompt=layer(self.visual_deep_prompts_parameters[int((index-1)/2)])
            

                self.compound_prompts_text[index]=text_prompt

                #self.compound_prompts_text[index] = layer(visual_deep_prompts[-1])
                #visual_prompt = self.compound_prompts_text[index]
                #visual_deep_prompts.append(visual_prompt)
                visual_deep_prompts[index]=self.visual_deep_prompts_parameters[int((index-1)/2)]
               # self.print_info(text_prompt, f"compound_prompts_text[{index}]")

        projected_ctx = self.proj_lang_to_vis(self.ctx)
     #   self.print_info(projected_ctx, "Projected ctx")

        return prompts, projected_ctx, self.compound_prompts_text, visual_deep_prompts

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.dtype = clip_model.dtype
        self.clip_model2 = clip_model

    def print_info(self, var, name):
        if isinstance(var, list):
            for i, v in enumerate(var):
                return
                self.print_info(v, f"{name}[{i}]")
        else:
            return
            print(f"{name}: shape={var.shape if isinstance(var, torch.Tensor) else 'N/A'}, dtype={var.dtype if isinstance(var, torch.Tensor) else type(var)}")

    def check_tensor_validity(self,tensor, name):
        if tensor is not None and isinstance(tensor, torch.Tensor):
            if torch.isnan(tensor).any():
                print(f"[DEBUG] NaN detected in {name}")
            if torch.isinf(tensor).any():
                print(f"[DEBUG] Inf detected in {name}")

    """def forward(self, image, label=None, caption=None, return_feature=False):  original forward function
        #print("caption", caption)
        tokenized_captions= clip.tokenize(caption).to("cuda") if caption else None
        with torch.no_grad():
            embedding_caption = self.clip_model2.token_embedding(tokenized_captions).type(self.dtype) if tokenized_captions is not None and tokenized_captions.numel() > 0 else None

        #if tokenized_captions is not None and tokenized_captions.numel() > 0:
            #print("Caption embedding shape", embedding_caption.size())
        tokenized_prompts = self.tokenized_prompts
        #self.print_info(tokenized_prompts, "tokenized_prompts")

        logit_scale = self.logit_scale.exp()
        #print(f"logit_scale: value={logit_scale.item()}")

        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()

        text_features = self.text_encoder(prompts, tokenized_prompts, deep_compound_prompts_text)
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision, embedding_caption)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        #print("Image feature dim", image_features.size())
        #print("Text feature dimensions", text_features.size())
        
        #if label is not None:
            #print("label is", label)
            #print("length of label",len(label))
            #print("label[0] is ",label[0])
            #print("text fature label[0]", text_features[label].size())
            #print()


        logits = logit_scale * image_features @ text_features.t()

        if self.prompt_learner.training:
            if label is not None and isinstance(label, torch.Tensor) and label.dtype == torch.float:
                log_probs = F.log_softmax(logits, dim=1)
                target_probs = label
                loss = F.kl_div(log_probs, target_probs, reduction='batchmean')
                text_features_for_images = label @ text_features                
            else:
                loss = F.cross_entropy(logits, label)
                text_features_for_images = text_features[label]
        
            alignment_loss = 1-F.cosine_similarity(image_features, text_features_for_images).mean()
            lambda_align = 0.5
            total_loss = loss + lambda_align * alignment_loss
            
            return total_loss
            # made changes from return logits
        if return_feature:
            return logits, image_features
        else:
            return logits"""
        

    def forward(self, image, label=None, caption=None, return_feature=False):
        # Sanitize caption input
        
        if caption is not None and isinstance(caption, list):
            if all(isinstance(c, str) for c in caption):  # Convert strings to tensors
                tokenized_captions = clip.tokenize(caption).to(image.device)
            elif all(isinstance(c, torch.Tensor) for c in caption):
                tokenized_captions = torch.stack(caption).to(image.device)
            else:
                raise ValueError("Invalid caption format")
        else:
            tokenized_captions = None

        with torch.no_grad():
            if tokenized_captions is not None and tokenized_captions.numel() > 0:
                embedding_caption = self.clip_model2.token_embedding(tokenized_captions).type(self.dtype)
                self.check_tensor_validity(embedding_caption, "embedding_caption")
            else:
                embedding_caption = None

        # Clamp logit scale to prevent explosion
        logit_scale = self.logit_scale.exp().clamp(max=100)  # Increased safety margin
        # print(f"[DEBUG] logit_scale: {logit_scale.item()}")

        # Get features with stability checks
        prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()
        
        # Text features
        text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_compound_prompts_text)
        self.check_tensor_validity(text_features, "text_features")
        
        # Image features
        image_features = self.image_encoder(image.type(self.dtype), shared_ctx, deep_compound_prompts_vision, embedding_caption)
        self.check_tensor_validity(image_features, "image_features")

        # Stable normalization with epsilon
        image_features = F.normalize(image_features, dim=-1, eps=1e-8)
        text_features = F.normalize(text_features, dim=-1, eps=1e-8)
        self.check_tensor_validity(image_features, "Normalized image_features")
        self.check_tensor_validity(text_features, "Normalized text_features")

        # Safe logit computation
        logits = logit_scale * torch.matmul(image_features, text_features.t())
        # print(f"[DEBUG] logits max: {logits.max().item()}, min: {logits.min().item()}")

        if self.training:
            # Add label sanity checks
            if label is not None:
                assert not torch.isnan(label).any(), "NaN in labels"
                assert label.max() < text_features.shape[0], "Label index out of bounds"

            # Compute loss with stability
            if label is not None and isinstance(label, torch.Tensor) and label.dtype == torch.float:
                log_probs = F.log_softmax(logits, dim=1)
                target_probs = label.clamp(min=1e-8)  # Prevent log(0)
                loss = F.kl_div(log_probs, target_probs, reduction="batchmean")
                text_features_for_images = label @ text_features
            else:
                loss = F.cross_entropy(logits, label)
                text_features_for_images = text_features[label]

            # Safe alignment loss
            cos_sim = F.cosine_similarity(image_features, text_features_for_images)
            self.check_tensor_validity(cos_sim, "cosine_similarity")
            alignment_loss = 1 - cos_sim.mean()
            
            # Combined loss
            lambda_align = 0.5
            total_loss = loss + lambda_align * alignment_loss
            
            # Final NaN check
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                raise RuntimeError("NaN/Inf in total loss")
                
            return total_loss
        #print("returning only")
        #print(label)
        return logits  # Return logits for evaluation
    

@TRAINER_REGISTRY.register()
class MaPLe(TrainerX):
    """
    Single-site trainer for the MaPLe approach, intended to be wrapped by
    a federated aggregator (MaPLeFederated) that merges label spaces.
    """
    def __init__(self, cfg, client_id=None, classnames=None):
        """
        If you're creating multiple MaPLe trainers, pass in a `client_id`
        so each trainer can register its model uniquely.
        """
        self.cfg = cfg
        self.client_id = client_id

        # NaN/inf tracking
        self.nan_count = 0
        self.total_batches = 0
        self.classnames= classnames
        super().__init__(cfg)

        # For debugging LR/grad norms
        self.lr_history = []
        self.grad_norms = []

        # After super().__init__(), the model is not yet built.
        # We'll build it below in build_model().

        # If you want to verify trainable params *after* build_model(),
        # do it at the end of build_model. Doing it here is before
        # the model actually exists.

    def check_cfg(self, cfg):
        """Check that precision config is valid."""
        assert cfg.TRAINER.MAPLE.PREC in ["fp16", "fp32", "amp"], (
            f"Invalid precision setting: {cfg.TRAINER.MAPLE.PREC}"
        )

    def build_model(self):
        """
        Build CLIP-based model with a prompt learner, 
        referencing the (already unified) .classnames from self.dm.
        """
        cfg = self.cfg

        # The aggregator has overwritten dataset.classnames to the unified list
        #classnames = self.dm.dataset.classnames
        classnames=self.classnames
        print(f"[Client {self.client_id}] # of classnames = {len(classnames)}")

        # 1) Load CLIP backbone from CPU (or similar)
        print(f"[Client {self.client_id}] Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        # Possibly switch to fp32 (or keep FP16 for speed)
        if cfg.TRAINER.MAPLE.PREC in ["fp32", "amp"]:
            clip_model.float()

        # 2) Build custom CLIP that includes the prompt learner
        print(f"[Client {self.client_id}] Building custom CLIP...")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        # 3) Freeze everything except certain layers
        print(f"[Client {self.client_id}] Turning off gradients except prompt_learner & LN/BN.")
        for param in self.model.parameters():
            param.requires_grad_(False)

        # Unfreeze LN/BN modules
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                for p in module.parameters():
                    p.requires_grad_(True)

        # Unfreeze prompt_learner
        prompt_key = "prompt_learner"
        for name, param in self.model.named_parameters():
            if prompt_key in name:
                param.requires_grad_(True)

        # Unfreeze any "VPT" parameters if you have Visual Prompt Tuning
        #for name, param in self.model.named_parameters():
            #if "VPT" in name:
                #param.requires_grad_(True)

        for name, param in self.model.named_parameters():
            if "visual.transformer.resblocks.11" in name:
                param.requires_grad_(True)

        # Or unfreeze final text encoder block
        for name, param in self.model.named_parameters():
            if "transformer.resblocks.11" in name:  # for text
                param.requires_grad_(True)
    
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                for p in module.parameters():
                    p.requires_grad_(True)

        # Double check
        enabled = set()
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                enabled.add(n)
        print(f"[Client {self.client_id}] Parameters to be updated: {enabled}")

        # 4) Optionally load pretrained weights (if provided in cfg)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        # 5) Move to device
        self.model.to(self.device)

        # 6) Build optimizer & scheduler
        # NOTE: you can filter so only prompt_learner parameters are given
        # but typically build_optimizer does that if you set up param_requires_grad properly.
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        # 7) Register model for Dassl's saving mechanism
        unique_key = f"MultiModalPromptLearner_{self.client_id}"
        if unique_key not in self._models:
            self.register_model(unique_key, self.model, self.optim, self.sched)

        # 8) Mixed precision scaler
        prec = cfg.TRAINER.MAPLE.PREC
        self.scaler = GradScaler() if prec == "amp" else None

        # 9) (Optional) DataParallel if multiple GPUs
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"[Client {self.client_id}] Multiple GPUs detected ({device_count}); using DataParallel.")
            self.model = nn.DataParallel(self.model)

        # 10) Summarize final # of trainable params
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Client {self.client_id}] Trainable params: {trainable_params:,}")

        # For debug, track initial LR
        self.lr_history = []
        initial_lr = self.optim.param_groups[0]['lr']
        self.lr_history.append(initial_lr)
        print(f"[Client {self.client_id}] Initial LR: {initial_lr}")

    def check_tensor_validity(self, tensor, name):
        """Helper to catch NaNs/Infs in inputs."""
        if tensor is None:
            raise ValueError(f"Null tensor: {name}")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Invalid tensor type: {name}")
        if torch.isnan(tensor).any():
            raise ValueError(f"NaN values in {name}")
        if torch.isinf(tensor).any():
            raise ValueError(f"Inf values in {name}")

    def parse_batch_train(self, batch):
        """Convert the incoming batch into the required inputs."""
        x = batch["img"]    # images
        y = batch["label"]  # labels
        c = batch["caption"]  # text or other meta
        x = x.to(self.device)
        y = y.to(self.device)
        # c might remain on CPU if it's text, depending on your usage
        return x, y, c

    def forward_backward(self, batch):
        """
        One forward & backward pass on a single batch.
        We do gradient clipping, handle AMP (if used), 
        and track LR changes.
        """
        image, label, caption = self.parse_batch_train(batch)
        self.total_batches += 1

        self.check_tensor_validity(image, "input image")
        self.check_tensor_validity(label, "input label")

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.MAPLE.PREC

        try:
            if prec == "amp":
                # Automatic Mixed Precision
                with autocast():
                    loss = model(image, label, caption)
                optim.zero_grad()
                scaler.scale(loss).backward()

                # Unscale before clipping
                scaler.unscale_(optim)

                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                    error_if_nonfinite=False  # We skip the error to handle non-finite gracefully
                )

                scaler.step(optim)
                scaler.update()

            else:
                # fp32 or (fp16 if you manually handle it)
                loss = model(image, label, caption)
                optim.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                    error_if_nonfinite=False
                )

                optim.step()

            # Track LR changes
            current_lr = optim.param_groups[0]['lr']
            if not self.lr_history or (current_lr != self.lr_history[-1]):
                self.lr_history.append(current_lr)
                print(f"[Client {self.client_id}] LR changed to: {current_lr}")

            # Compute grad norm for logging (optional)
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            self.grad_norms.append(total_norm)

            return {"loss": loss.item()}

        except RuntimeError as e:
            # If non-finite gradients occur
            if 'non-finite' in str(e).lower():
                self.nan_count += 1
                nan_rate = self.nan_count / self.total_batches
                print(f"[Client {self.client_id}] NaN rate: {nan_rate:.2%}")
                print(f"Non-finite gradients detected at batch_idx={self.batch_idx}. Skipping batch.")
                optim.zero_grad()  # Reset the optimizer to avoid corrupted state
                return {"loss": float('nan')}
            else:
                raise  # re-raise if it's not a non-finite error

    def run_epoch(self, epoch):
        """
        Called by aggregator for each local epoch.
        This iterates over our train_loader once.
        """
        self.model.train()
        total_loss = 0.0
        total_steps = 0

        for batch_idx, batch in enumerate(self.dm.train_loader):
            self.batch_idx = batch_idx
            loss_dict = self.forward_backward(batch)  # e.g. {"loss": loss_value}
            if "loss" in loss_dict:
                total_loss += loss_dict["loss"]
            total_steps += 1

        self.update_lr()
        local_eval = self.test()  # or some validation method
        local_acc = local_eval["accuracy"] if "accuracy" in local_eval else 0


        # return average epoch loss
        avg_loss = total_loss / max(1, total_steps)
        print(f"[Client {self.client_id}] Epoch {epoch} done. Loss={avg_loss:.4f}, Acc={local_acc:.2f}%")
        return {"avg_loss": avg_loss}

    def update_lr(self):
        """Step the LR scheduler at epoch's end (if configured)."""
        if self.sched is not None:
            self.sched.step()

    def test(self, evaluate_train=False):
        """
        Simple test method on our test set. 
        Returns a dict with the final accuracy.
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            loader = self.dm.test_loader
            for batch in loader:
                x, y, c = self.parse_batch_train(batch)
                # Forward pass for classification
                outputs = self.model(x)  # presumably returns logits
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = (100.0 * correct / total) if total > 0 else 0.0
        print(f"[Client {self.client_id}] Test Accuracy: {acc:.2f}%")
        return {"accuracy": acc}

    def load_model(self, directory, epoch=None):
        """
        If needed: load local weights from a checkpoint on disk.
        Usually you'd rely on aggregator's broadcast instead.
        """
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, load "model-best.pth.tar"
        model_file = "model-best.pth.tar"
        if epoch is not None:
            model_file = f"model.pth.tar-{epoch}"

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f"Model not found at '{model_path}'")

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            loaded_epoch = checkpoint["epoch"]

            # If you want to ignore certain prompt vectors:
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]
            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print(f"[Client {self.client_id}] Loading weights into {name} "
                  f"from '{model_path}' (epoch={loaded_epoch})")
            self._models[name].load_state_dict(state_dict, strict=False)