import argparse
import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer

# custom
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet

import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r
import datasets.patternnet
import datasets.ucmerced
import datasets.resics
# import datasets.patternnet # Duplicate import removed
import trainers.coop
import trainers.cocoop
import trainers.zsclip
import trainers.maple
import trainers.independentVL
import trainers.vpt
import trainers.maple_fed  # fed change 1
import trainers.maple_fed_tester  # for single-dataset testing


def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    # Removed resume from here as it's not used in test script
    # if args.resume:
    #     cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    # Config for MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = 2  # number of context vectors
    cfg.TRAINER.MAPLE.CTX_INIT = "a satellite image of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 3 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"
      # all, base or new

    # Config for independent Vision Language prompting (independent-vlp)
    cfg.TRAINER.IVLP = CN()
    cfg.TRAINER.IVLP.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.IVLP.N_CTX_TEXT = 2  # number of context vectors at the language branch
    cfg.TRAINER.IVLP.CTX_INIT = "a photo of a"  # initialization words (only for language prompts)
    cfg.TRAINER.IVLP.PREC = "fp16"  # fp16, fp32, amp
    # If both variables below are set to 0, 0, will the config will degenerate to COOP model
    cfg.TRAINER.IVLP.PROMPT_DEPTH_VISION = 9 # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.TRAINER.IVLP.PROMPT_DEPTH_TEXT = 9  # Max 12, minimum 0, for 0 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new

    # Config for only vision side prompting
    cfg.TRAINER.VPT = CN()
    cfg.TRAINER.VPT.N_CTX_VISION = 2  # number of context vectors at the vision branch
    cfg.TRAINER.VPT.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.VPT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.VPT.PROMPT_DEPTH_VISION = 1  # if set to 1, will represent shallow vision prompting only
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
# fed change 2
    cfg.FED = CN()
    cfg.FED.NUM_CLIENTS = 2
    cfg.FED.NUM_ROUNDS = 30
    cfg.FED.LOCAL_EPOCHS = 10

def generate_and_save_tsne_plot(trainer, cfg, args):
    """Generates a t-SNE plot from model embeddings."""
    print("Generating t-SNE plot...")

    # 1. Determine model and data loader to use
    if hasattr(trainer, 'clients') and len(trainer.clients) > 0:
        print("Using client 0's model for embedding extraction")
        model = trainer.clients[0].model
        data_loader = trainer.client_data_managers[0].test_loader
        class_names_map = trainer.lab2cname
    elif hasattr(trainer, 'model'):
        model = trainer.model
        data_loader = trainer.test_loader
        class_names_map = {i: name for i, name in enumerate(trainer.dm.dataset.classnames)} if hasattr(trainer.dm.dataset, 'classnames') else {}
    else:
        print("ERROR: Could not determine model structure.")
        return

    # 2. Extract features
    print(f"Extracting features using device: {trainer.device}")
    all_features = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            images = batch["img"].to(trainer.device)
            labels = batch["label"].cpu().numpy()
            
            # Simple approach - use full model forward pass with return_feature
            if hasattr(model, 'prompt_learner'):
                print("Using MaPLe model extraction method")
                try:
                    logits, features = model(images, return_feature=True)
                except:
                    print("Could not extract features with return_feature, skipping t-SNE")
                    return
            else:
                # Standard approach for non-MaPLe models
                print("Using standard model extraction method")
                image_encoder = model.image_encoder if hasattr(model, 'image_encoder') else model.visual
                features = image_encoder(images)
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels)
            
            if sum(len(f) for f in all_features) >= args.tsne_max_samples:
                print(f"Reached max samples for t-SNE ({args.tsne_max_samples}).")
                break
    
    # 3. Process features and run t-SNE
    if not all_features:
        print("No features extracted. Skipping t-SNE plot.")
        return
    
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Subsample if needed
    if len(all_features) > args.tsne_max_samples:
        indices = np.random.choice(len(all_features), args.tsne_max_samples, replace=False)
        all_features = all_features[indices]
        all_labels = all_labels[indices]
    
    # Run t-SNE
    print(f"Running t-SNE on {len(all_features)} samples...")
    tsne = TSNE(
        n_components=2,
        perplexity=args.tsne_perplexity,
        max_iter=args.tsne_n_iter,
        random_state=cfg.SEED if cfg.SEED >= 0 else None
    )
    tsne_results = tsne.fit_transform(all_features)
    
    # 4. Calculate metrics (if possible)
    silhouette_avg = "N/A"
    unique_labels = np.unique(all_labels)
    if len(unique_labels) > 1:
        try:
            silhouette_avg = f"{silhouette_score(all_features, all_labels):.4f}"
        except Exception as e:
            print(f"Could not calculate Silhouette Score: {e}")
    
    # 5. Create the plot
    plt.figure(figsize=(12, 10))
    
    # Generate colors for classes
    cmap = plt.get_cmap('tab10' if len(unique_labels) <= 10 else 'tab20')
    colors = [cmap(i % cmap.N) for i in range(len(unique_labels))]
    
    # Plot each class
    for i, label in enumerate(unique_labels):
        mask = all_labels == label
        plt.scatter(
            tsne_results[mask, 0],
            tsne_results[mask, 1],
            color=colors[i],
            alpha=0.7,
            s=50,
            label=class_names_map.get(int(label), f"Class {label}") if class_names_map else f"Class {label}"
        )
    
    # Add title and legend
    dataset_name = cfg.DATASET.NAME if hasattr(cfg, 'DATASET') and hasattr(cfg.DATASET, 'NAME') else 'Unknown'
    plt.title(f"t-SNE Plot: {dataset_name}\nSilhouette Score: {silhouette_avg}", fontsize=14)
    
    if len(unique_labels) <= 20:  # Only show legend if not too many classes
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.figtext(0.5, 0.02, "Our model", ha="center", fontsize=24, color='gray')
    plt.tight_layout()
    
    # 6. Save the plot
    output_dir = os.path.dirname(args.tsne_output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(args.tsne_output_file, bbox_inches='tight', dpi=150)
    print(f"t-SNE plot saved to {args.tsne_output_file}")
    
    # Save high-resolution version if requested
    if hasattr(args, 'tsne_high_res') and args.tsne_high_res:
        high_res_path = args.tsne_output_file.replace('.png', '_high_res.png')
        print(f"Saving high-resolution version to: {high_res_path}")
        plt.savefig(high_res_path, bbox_inches='tight', dpi=300)
        print(f"High-resolution t-SNE plot saved to: {high_res_path}")
    plt.close()


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    # Directly load model and test
    # args.model_dir and args.load_epoch are required by argparse now
    trainer.load_model(args.model_dir, epoch=args.load_epoch)
    
    if args.generate_tsne_plot:
        generate_and_save_tsne_plot(trainer, cfg, args)
        
        # High-res version is now handled within the generate_and_save_tsne_plot function
        # to avoid matplotlib state issues

    trainer.test()
    print("Testing complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a pre-trained model on a specified dataset.")
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory for logs and results")
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG (if applicable)"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG (if applicable)"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to method config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer (e.g. CoOp, MaPLe)")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="directory of the pre-trained model to load",
    )
    parser.add_argument(
        "--load-epoch",
        type=int,
        required=True,
        help="epoch of the pre-trained model to load",
    )
    parser.add_argument(
        "--generate-tsne-plot",
        action="store_true",
        help="Generate and save a t-SNE plot of feature embeddings.",
    )
    parser.add_argument(
        "--tsne-output-file",
        type=str,
        default="tsne_plot.png",
        help="Path to save the t-SNE plot. Used if --generate-tsne-plot is active. Can include directories.",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="Perplexity for t-SNE. Used if --generate-tsne-plot is active.",
    )
    parser.add_argument(
        "--tsne-n-iter",
        type=int,
        default=1000,
        help="Maximum number of iterations for t-SNE (max_iter parameter). Used if --generate-tsne-plot is active.",
    )
    parser.add_argument(
        "--tsne-max-samples",
        type=int,
        default=2000,
        help="Maximum number of samples to use for t-SNE from the test set. Used if --generate-tsne-plot is active.",
    )
    parser.add_argument(
        "--tsne-high-res",
        action="store_true",
        help="Save a high-resolution version of the t-SNE plot.",
    )
    parser.add_argument(
        "--tsne-individual-plots",
        action="store_true",
        help="Save individual class plots for detailed analysis.",
    )
    parser.add_argument(
        "--tsne-show-hulls",
        action="store_true",
        help="Show convex hulls around clusters with sufficient points.",
    )
    parser.add_argument(
        "--tsne-annotate-confidence",
        action="store_true",
        help="Annotate classes with confidence scores and convex hulls.",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line (e.g. DATASET.NAME OxfordPets)",
    )
    args = parser.parse_args()
    main(args) 

# Example command for generating t-SNE plot with all options:
# 
# python test_model.py \
#   --model-dir /path/to/model/checkpoint/directory \
#   --load-epoch 2 \
#   --config-file configs/trainers/MaPLeFed/resisc45.yaml \
#   --trainer MaPLeFederatedTester \
#   --generate-tsne-plot \
#   --tsne-output-file output/tsne_plots/resisc45_tsne.png \
#   --tsne-perplexity 40.0 \
#   --tsne-n-iter 2000 \
#   --tsne-max-samples 3000 \
#   --tsne-high-res \
#   --tsne-individual-plots \
#   --tsne-show-hulls \
#   --tsne-annotate-confidence \
#   DATASET.NAME RESICS 