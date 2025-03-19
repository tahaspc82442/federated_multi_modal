# run_fedclip.py
import argparse
import torch
from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
#import wandb

# Import your custom trainer *and* the necessary datasets.  This is
# important because DASSL needs to be able to *find* your trainer
# class when you specify it by name.  The dataset imports are needed
# even though you're overriding the data loading.
import trainers.fedclip_federated
import datasets.oxford_pets
import datasets.milaid
import datasets.patternnet
import datasets.mlrs
import datasets.ucmerced
import datasets.eurosat


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
    if args.resume:
        cfg.RESUME = args.resume
    if args.seed:
        cfg.SEED = args.seed
    if args.trainer:
        cfg.TRAINER.NAME = args.trainer
    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone
    # Add any other argument-to-config mappings here

def extend_cfg(cfg):
    """Add custom configuration nodes.  Crucially, this includes your FED node."""
    from yacs.config import CfgNode as CN

    cfg.TRAINER.FEDCLIPFEDERATED = CN() # Add a node for your trainer's specific settings
    cfg.FED = CN()
    cfg.FED.NUM_CLIENTS = 5
    cfg.FED.NUM_ROUNDS = 30
    cfg.FED.LOCAL_EPOCHS = 10
    cfg.FED.EVAL_FREQ = 5
    cfg.MODEL.DEVICE = "cuda"
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"

# def setup_cfg(args):
#     cfg = get_cfg_default()
#     extend_cfg(cfg)

#     # 1. From the dataset config file (required by DASSL)
#     if args.dataset_config_file:
#         cfg.merge_from_file(args.dataset_config_file)

#     # 2. From the method config file (your FedCLIP config)
#     if args.config_file:
#         cfg.merge_from_file(args.config_file)

#     # 3. From input arguments
#     reset_cfg(cfg, args)

#     # 4. From optional input arguments (the --opts list)
#     cfg.merge_from_list(args.opts)

#     cfg.freeze()
#     return cfg

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)  # Make sure this is called *before* merging

    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    if args.config_file:
        try:  # Add a try-except block
            cfg.merge_from_file(args.config_file)
        except KeyError as e:  # Catch the KeyError
            print(f"ERROR: Missing configuration key: {e}")  # Print the full error
            raise  # Re-raise the exception to stop execution

    reset_cfg(cfg, args)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def main(args):
    cfg = setup_cfg(args)

    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    # wandb.init(
    #     project="fedclip_project",  # Replace with your project name
    #     entity="your_wandb_entity", # Replace with your WandB entity
    #     name=args.run_name,
    #     config=cfg,
    # )

    trainer = build_trainer(cfg)

    if args.eval_only:
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test()
        return

    if not args.no_train:
        trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="", help="Path to dataset root")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    parser.add_argument("--config-file", type=str, required=True, help="Path to the trainer config file")
    parser.add_argument("--dataset-config-file", type=str, required=True, help="Path to a dataset config file")
    parser.add_argument("--trainer", type=str, required=True, help="Name of the trainer (FedCLIPFederated)")
    parser.add_argument("--backbone", type=str, default="ViT-B/32", help="CLIP backbone")
    parser.add_argument("--eval-only", action="store_true", help="Only perform evaluation")
    parser.add_argument("--model-dir", type=str, default="", help="Directory to load model from for eval")
    parser.add_argument("--load-epoch", type=int, help="Epoch to load for evaluation")
    parser.add_argument("--no-train", action="store_true", help="Do not train")
    parser.add_argument("--run-name", type=str, default="fedclip_run", help="WandB run name")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="Modify config options")

    args = parser.parse_args()
    main(args)