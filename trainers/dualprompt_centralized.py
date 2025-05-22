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
    design_details = {"trainer": 'DualPromptCentralized',
                      "vision_depth": 0,
                      "language_depth": 0, "vision_ctx": 0,
                      "language_ctx": 0,
                      "maple_length": cfg.TRAINER.DUALPROMPTCENTRALIZED.N_CTX}
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

class DualPromptCentralizedPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.DUALPROMPTCENTRALIZED.N_CTX
        ctx_init = cfg.TRAINER.DUALPROMPTCENTRALIZED.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg.TRAINER.DUALPROMPTCENTRALIZED.PROMPT_DEPTH >= 1, "For DualPromptCentralized, PROMPT_DEPTH should be >= 1"
        
        self.compound_prompts_depth = cfg.TRAINER.DUALPROMPTCENTRALIZED.PROMPT_DEPTH
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
        print('DualPromptCentralized design: Dual-modal Prompt Learning')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of DualPromptCentralized context words (tokens): {n_ctx}")

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
        self.prompt_learner = DualPromptCentralizedPromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
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

    def forward(self, image, label=None, caption=None, return_feature=False):
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

        # print("Image feature dim", image_features.size())
        # print("Text feature dimensions", text_features.size())
        
        # if label is not None:
        #     print("label is", label)
        #     print("length of label",len(label))
        #     print("label[0] is ",label[0])
        #     print("text fature label[0]", text_features[label].size())


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
            return logits
        






def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


@TRAINER_REGISTRY.register()
class DualPromptCentralized(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.DUALPROMPTCENTRALIZED.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.DUALPROMPTCENTRALIZED.PREC == "fp32" or cfg.TRAINER.DUALPROMPTCENTRALIZED.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for param in self.model.parameters():
            param.requires_grad_(False)

        for name, module in self.model.named_modules():
            if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
                for param in module.parameters():
                    param.requires_grad_(True)

        for name, param in self.model.named_parameters():
            if name_to_update in name:
                param.requires_grad_(True)

        for name, param in self.model.named_parameters():
            if "VPT" in name:
                param.requires_grad_(True)


        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("DualPromptCentralized", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.DUALPROMPTCENTRALIZED.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label , caption = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.DUALPROMPTCENTRALIZED.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label, caption)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label, caption)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        caption = batch["caption"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label, caption

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)