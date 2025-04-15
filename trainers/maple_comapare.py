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
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        combined = [x, compound_prompts_deeper_text, 0]
        outputs = self.transformer(combined)
        x = outputs[0]
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.MAPLE.N_CTX
        ctx_init = cfg.TRAINER.MAPLE.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        device = next(clip_model.parameters()).device

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg.TRAINER.MAPLE.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.MAPLE.PROMPT_DEPTH
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype, device=device)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('MaPLe design: Multi-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of MaPLe context words (tokens): {n_ctx}")

        self.proj_lang_to_vis = nn.Linear(ctx_dim, 768).to(dtype=dtype)
        self.proj_vis_to_lang = nn.Linear(768, ctx_dim).to(dtype=dtype)
        self.ctx = nn.Parameter(ctx_vectors)

        self.compound_prompts_text_parameters = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512, dtype=dtype, device=device))
                                                      for i in range(self.compound_prompts_depth - 1) if i%2==0])
        self.visual_deep_prompts_parameters = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 768, device=device, dtype=dtype)) for i in range(self.compound_prompts_depth - 1) if i%2!=0])

        for single_para in self.compound_prompts_text_parameters:
            nn.init.normal_(single_para, std=0.02)
        for single_para in self.visual_deep_prompts_parameters:
            nn.init.normal_(single_para, std=0.02)

        self.compound_prompt_projections = nn.ModuleList([nn.Linear(ctx_dim, 768) if i % 2 == 0
                                                          else nn.Linear(768, ctx_dim)
                                                          for i in range(self.compound_prompts_depth-1)])
        self.compound_prompts_text=[0 for i in range(self.compound_prompts_depth-1)]


        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p).to(device) for p in prompts])  # (n_cls, n_tkn)
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
                prefix,
                ctx,
                suffix,
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

        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        prompts = self.construct_prompts(ctx, prefix, suffix)

        visual_deep_prompts = [0 for i in range(self.compound_prompts_depth-1)]

        for index, layer in enumerate(self.compound_prompt_projections):
            if index % 2 == 0:
                visual_prompt = layer(self.compound_prompts_text_parameters[int(index/2)])
                visual_deep_prompts[index] = visual_prompt
                self.compound_prompts_text[index] = self.compound_prompts_text_parameters[int(index/2)]
            else:
                text_prompt=layer(self.visual_deep_prompts_parameters[int((index-1)/2)])
                self.compound_prompts_text[index]=text_prompt
                visual_deep_prompts[index]=self.visual_deep_prompts_parameters[int((index-1)/2)]

        projected_ctx = self.proj_lang_to_vis(self.ctx)

        return prompts, projected_ctx, self.compound_prompts_text, visual_deep_prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        print(len(classnames))
        print("these classnames clip got", classnames)
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

    def forward(self, image, label=None, caption=None, return_feature=False):
        with autocast(enabled=(self.dtype == torch.float16)):
            if caption is not None and isinstance(caption, list):
                if all(isinstance(c, str) for c in caption):
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

            logit_scale = self.logit_scale.exp().clamp(max=100)

            prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = self.prompt_learner()

            text_features = self.text_encoder(prompts.to(dtype =self.dtype), self.tokenized_prompts,[t.to(dtype=self.dtype) for t in deep_compound_prompts_text])
            self.check_tensor_validity(text_features, "text_features")

            image_features = self.image_encoder(
                image.type(self.dtype),
                shared_ctx.to(dtype=self.dtype),
                [v.to(dtype=self.dtype) for v in deep_compound_prompts_vision],
                embedding_caption
            )
            self.check_tensor_validity(image_features, "image_features")

            image_features = F.normalize(image_features, dim=-1, eps=1e-8)
            text_features = F.normalize(text_features, dim=-1, eps=1e-8)
            self.check_tensor_validity(image_features, "Normalized image_features")
            self.check_tensor_validity(text_features, "Normalized text_features")

            logits = logit_scale * torch.matmul(image_features, text_features.t())

            if self.training:
                if label is not None:
                    assert not torch.isnan(label).any(), "NaN in labels"
                    assert label.max() < text_features.shape[0], "Label index out of bounds"

                if label is not None and isinstance(label, torch.Tensor) and label.dtype == torch.float:
                    log_probs = F.log_softmax(logits, dim=1)
                    target_probs = label.clamp(min=1e-8)
                    loss = F.kl_div(log_probs, target_probs, reduction="batchmean")
                    text_features_for_images = label @ text_features
                else:
                    loss = F.cross_entropy(logits, label)
                    text_features_for_images = text_features[label]

                cos_sim = F.cosine_similarity(image_features, text_features_for_images)
                self.check_tensor_validity(cos_sim, "cosine_similarity")
                alignment_loss = 1 - cos_sim.mean()

                lambda_align = 0.5
                total_loss = loss + lambda_align * alignment_loss

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    raise RuntimeError("NaN/Inf in total loss")

                return total_loss

            return logits


@TRAINER_REGISTRY.register()
class MaPLe(TrainerX):

    def __init__(self, cfg, client_id=None, classnames=None, _clip_model = None):

        self.cfg = cfg
        self.client_id = client_id
        self._clip_model = _clip_model
        self.nan_count = 0
        self.total_batches = 0
        self.classnames= classnames

        self.global_model_params_start_round = None
        self.prox_mu = 0.1

        super().__init__(cfg)

        self.lr_history = []
        self.grad_norms = []


    def check_cfg(self, cfg):
        assert cfg.TRAINER.MAPLE.PREC in ["fp16", "fp32", "amp"], (
            f"Invalid precision setting: {cfg.TRAINER.MAPLE.PREC}"
        )

    def configure_trainable_params(self, model, freeze_deep_layers=False):
        PROMPT_TOKENS = ['ctx', 'prompts_parameters', 'compound_prompts']
        NORM_LAYERS = ['ln_', 'layer_norm', 'ln_pre', 'ln_post']
        ATTN_POOL_PATTERNS = ['attention_weights']
        DEEP_LAYERS = {
            'visual': [8, 9, 10, 11],
            'transformer': [9, 10, 11]
        }

        print(f"Configuring trainable parameters... Freeze deep layers: {freeze_deep_layers}")
        total_params = 0
        trainable_params = 0

        for name, param in model.named_parameters():
            total_params += param.numel()
            param.requires_grad = False

            is_norm = any(nl in name for nl in NORM_LAYERS)
            is_prompt = any(pt in name for pt in PROMPT_TOKENS)
            is_attn_pool = any(ap in name for ap in ATTN_POOL_PATTERNS)
            is_deep = False

            if 'resblocks' in name:
                parts = name.split('.')
                try:
                    resblocks_idx = parts.index('resblocks')
                    if resblocks_idx + 1 < len(parts) and parts[resblocks_idx + 1].isdigit():
                        block_num = int(parts[resblocks_idx + 1])
                        encoder_type = None
                        if 'visual' in parts: encoder_type = 'visual'
                        elif 'transformer' in parts: encoder_type = 'transformer'
                        if encoder_type and encoder_type in DEEP_LAYERS:
                            is_deep = block_num in DEEP_LAYERS[encoder_type]
                except (ValueError, IndexError):
                    pass

            if is_prompt:
                param.requires_grad = True
            elif is_attn_pool:
                param.requires_grad = True
            elif is_norm:
                if not (freeze_deep_layers and is_deep):
                    param.requires_grad = True

            if param.requires_grad:
                trainable_params += param.numel()

        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print("-" * 30)


    def build_model(self):
        cfg = self.cfg
        classnames=self.classnames
        print(f"[Client {self.client_id}] # of classnames = {len(classnames)}")

        print(f"[Client {self.client_id}] Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model =self._clip_model if self._clip_model is not None else load_clip_to_cpu(cfg)

        if cfg.TRAINER.MAPLE.PREC in ["fp32", "amp"]:
            clip_model.float()

        print(f"[Client {self.client_id}] Building custom CLIP...")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print(f"[Client {self.client_id}] Turning off gradients except prompt_learner & LN/BN.")

        self.configure_trainable_params(self.model, freeze_deep_layers=False)

        enabled = set()
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                enabled.add(n)
        print(f"[Client {self.client_id}] Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)

        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        unique_key = f"MultiModalPromptLearner_{self.client_id}"
        if unique_key not in self._models:
            self.register_model(unique_key, self.model, self.optim, self.sched)

        prec = cfg.TRAINER.MAPLE.PREC
        self.scaler = GradScaler() if prec == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"[Client {self.client_id}] Multiple GPUs detected ({device_count}); using DataParallel.")
            self.model = nn.DataParallel(self.model)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[Client {self.client_id}] Trainable params: {trainable_params:,}")

        self.lr_history = []
        initial_lr = self.optim.param_groups[0]['lr']
        self.lr_history.append(initial_lr)
        print(f"[Client {self.client_id}] Initial LR: {initial_lr}")

    def check_tensor_validity(self, tensor, name):
        if tensor is None:
            raise ValueError(f"Null tensor: {name}")
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Invalid tensor type: {name}")
        if torch.isnan(tensor).any():
            raise ValueError(f"NaN values in {name}")
        if torch.isinf(tensor).any():
            raise ValueError(f"Inf values in {name}")

    def parse_batch_train(self, batch):
        x = batch["img"]
        y = batch["label"]
        c = batch["caption"]
        x = x.to(self.device)
        y = y.to(self.device)
        return x, y, c


    def forward_backward(self, batch):
        image, label, caption = self.parse_batch_train(batch)
        self.total_batches += 1

        self.check_tensor_validity(image, "input image")
        self.check_tensor_validity(label, "input label")

        model = self.model
        optim = self.optim
        scaler = self.scaler
        prec = self.cfg.TRAINER.MAPLE.PREC

        original_loss = None
        total_loss = None
        proximal_term = torch.tensor(0.0, device=self.device)

        try:
            if prec == "amp":
                with autocast():
                    original_loss = model(image, label, caption)
            else:
                original_loss = model(image, label, caption)

            if original_loss is None:
                 raise RuntimeError("Model did not return a loss value.")
            if not isinstance(original_loss, torch.Tensor):
                 raise TypeError(f"Model returned type {type(original_loss)}, expected torch.Tensor.")


            if self.prox_mu > 0 and self.global_model_params_start_round is not None:
                prox_term_sum = torch.tensor(0.0, device=self.device)
                current_params = {name: param for name, param in model.named_parameters() if param.requires_grad}

                for name, local_param in current_params.items():
                    if name in self.global_model_params_start_round:
                        global_param = self.global_model_params_start_round[name].to(local_param.device)

                        if local_param.shape == global_param.shape:
                            prox_term_sum += torch.norm(local_param - global_param, p=2)**2
                        else:
                            print(f"[Client {self.client_id} Warn] FedProx: Shape mismatch for '{name}'. Global: {global_param.shape}, Local: {local_param.shape}. Skipping param.")

                proximal_term = (0.5 * self.prox_mu) * prox_term_sum

            total_loss = original_loss + proximal_term

            optim.zero_grad()

            if prec == "amp":
                scaler.scale(total_loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                    error_if_nonfinite=False
                )
                scaler.step(optim)
                scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                    error_if_nonfinite=False
                )
                optim.step()

            current_lr = optim.param_groups[0]['lr']
            if not hasattr(self, 'lr_history'): self.lr_history = []
            if not self.lr_history or (current_lr != self.lr_history[-1]):
                self.lr_history.append(current_lr)
                print(f"[Client {self.client_id}] LR changed to: {current_lr}")

            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            if not hasattr(self, 'grad_norms'): self.grad_norms = []
            self.grad_norms.append(total_norm)

            return {"loss": total_loss.item(), "original_loss": original_loss.item()}

        except RuntimeError as e:
            if not hasattr(self, 'nan_count'): self.nan_count = 0

            if 'non-finite' in str(e).lower() or 'inf' in str(e).lower() or 'nan' in str(e).lower():
                self.nan_count += 1
                nan_rate = self.nan_count / self.total_batches if self.total_batches > 0 else 0
                batch_info = f"batch_idx={self.batch_idx}" if hasattr(self, 'batch_idx') else "current batch"
                print(f"[Client {self.client_id}] Warning: Non-finite gradients detected at {batch_info}. Skipping batch. Error: {e}")
                print(f"[Client {self.client_id}] NaN/Inf rate: {nan_rate:.2%}")
                optim.zero_grad()
                return {"loss": float('nan'), "original_loss": float('nan')}
            else:
                 batch_info = f"batch_idx={self.batch_idx}" if hasattr(self, 'batch_idx') else "current batch"
                 print(f"[Client {self.client_id}] Error during forward/backward at {batch_info}: {e}")
                 raise

    def set_global_model_params(self, global_model_state_dict):
        print(f"[Client {self.client_id}] Storing global model parameters for FedProx.")
        self.global_model_params_start_round = {
            name: param.clone().detach() for name, param in global_model_state_dict.items()
        }

    def run_epoch(self, epoch):
        self.model.train()
        total_original_loss = 0.0
        total_combined_loss = 0.0
        valid_steps = 0

        if self.dm.train_loader is None:
             print(f"[Client {self.client_id}] Error: train_loader is None.")
             return {"avg_loss": 0.0, "accuracy": 0.0}

        for batch_idx, batch in enumerate(self.dm.train_loader):
            self.batch_idx = batch_idx

            loss_dict = self.forward_backward(batch)

            original_loss_val = loss_dict.get("original_loss", float('nan'))
            combined_loss_val = loss_dict.get("loss", float('nan'))

            if loss_dict is not None and not math.isnan(original_loss_val):
                total_original_loss += original_loss_val
                if not math.isnan(combined_loss_val):
                    total_combined_loss += combined_loss_val
                valid_steps += 1
            else:
                 print(f"[Client {self.client_id}] Warning: Skipping batch {batch_idx} in epoch {epoch} due to NaN loss.")

        if hasattr(self, 'update_lr') and callable(self.update_lr):
             self.update_lr()

        print(f"[Client {self.client_id}] Epoch {epoch} Training Done ({valid_steps} valid steps). Running local evaluation...")
        local_eval = {}
        if hasattr(self, 'test') and callable(self.test):
            try:
                with torch.no_grad():
                    local_eval = self.test()
            except Exception as e:
                 print(f"[Client {self.client_id}] Error during local evaluation: {e}")
                 local_eval = {}
        else:
             print(f"[Client {self.client_id}] Warning: self.test() method not found. Skipping local evaluation.")

        local_acc = local_eval.get("accuracy", 0.0)

        avg_original_loss = total_original_loss / max(1, valid_steps)
        avg_combined_loss = total_combined_loss / max(1, valid_steps)

        print(f"[Client {self.client_id}] Epoch {epoch} Result: Avg Original Loss={avg_original_loss:.4f}, "
              f"(Avg Combined Loss={avg_combined_loss:.4f}), Local Acc={local_acc:.2f}%")

        return {"avg_loss": avg_original_loss, "accuracy": local_acc}

    def update_lr(self):
        if self.sched is not None:
            self.sched.step()

    def test(self, evaluate_train=False):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            loader = self.dm.test_loader
            for batch in loader:
                x, y, c = self.parse_batch_train(batch)

                outputs = self.model(x)
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        acc = (100.0 * correct / total) if total > 0 else 0.0
        return {"accuracy": acc}

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

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

            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]
            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print(f"[Client {self.client_id}] Loading weights into {name} "
                  f"from '{model_path}' (epoch={loaded_epoch})")
            self._models[name].load_state_dict(state_dict, strict=False)