""""Train fubction modified to work with wandb sweep"""


import argparse
import torch

from dassl.utils import setup_logger, set_random_seed, collect_env_info
from dassl.config import get_cfg_default
from dassl.engine import build_trainer
import wandb
   
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
import datasets.mlrs
import datasets.milaid
import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r
import datasets.patternnet
import datasets.ucmerced
import datasets.patternnet
import trainers.coop
import trainers.cocoop
import trainers.zsclip
import trainers.maple
import trainers.independentVL
import trainers.vpt
import trainers.maple_fed  # fed change 1
import logging 

logger = logging.getLogger(__name__)

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
    cfg.TRAINER.MAPLE.CTX_INIT = "a satellite image of a" #"a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 6 #changed from 9 to 2 # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.TRAINER.MAPLE.LAMBDA_ALIGN = 0.1    # lambda for alignment loss

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
    cfg.FED.NUM_CLIENTS = 10
    cfg.FED.NUM_ROUNDS = 50
    cfg.FED.LOCAL_EPOCHS = 10
    cfg.PROX_MU= 0.0
    cfg.FED.NUM_PARTITIONS_PER_DATASET = 10

def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg) # Defines custom params like LAMBDA_ALIGN

    # 1. From the dataset config file
    if args.dataset_config_file:
        logger.info(f"Merging config from dataset file: {args.dataset_config_file}")
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        logger.info(f"Merging config from method file: {args.config_file}")
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments (like --seed, --output-dir etc.)
    reset_cfg(cfg, args)

    # 4. Preprocess and merge optional input arguments from args.opts
    #    Handles both 'KEY VALUE' and '--KEY=VALUE' formats.
    if args.opts:
        logger.info(f"Processing command line opts: {args.opts}")
        processed_opts = []
        i = 0
        while i < len(args.opts):
            opt = args.opts[i]
            if opt.startswith("--"):
                # Handle '--KEY=VALUE' format (from wandb agent ${args})
                if "=" in opt:
                    key_value = opt.lstrip("-").split("=", 1)
                    if len(key_value) == 2:
                        key, value = key_value
                        processed_opts.extend([key, value])
                        logger.debug(f"  Processed '--key=value': {key} = {value}")
                        i += 1
                    else:
                        logger.warning(f"  Skipping incorrectly formatted '--key=value' argument: {opt}")
                        i += 1
                else:
                    # Handle '--key' potentially followed by 'value' (less common for opts, but possible)
                    # Or flags like '--eval-only' if they accidentally end up in opts
                    key = opt.lstrip("-")
                    # Check if next arg exists and doesn't start with '-' (likely the value)
                    if i + 1 < len(args.opts) and not args.opts[i+1].startswith("-"):
                        value = args.opts[i+1]
                        processed_opts.extend([key, value])
                        logger.debug(f"  Processed '--key value': {key} = {value}")
                        i += 2
                    else:
                        # Treat as a boolean flag maybe? Or just skip? Skipping is safer.
                        logger.warning(f"  Skipping flag-like argument or key without value found in opts: {opt}")
                        i += 1
            else:
                # Handle 'KEY VALUE' format
                key = opt
                if i + 1 < len(args.opts):
                    value = args.opts[i+1]
                    processed_opts.extend([key, value])
                    logger.debug(f"  Processed 'key value': {key} = {value}")
                    i += 2
                else:
                    logger.error(f"  Error: Option key '{key}' found at the end of opts list is missing a value.")
                    # Decide whether to raise error or just warn and continue
                    # raise ValueError(f"Option key '{key}' needs a value.")
                    i += 1 # Increment to avoid infinite loop

        # Final check: Ensure the processed list has pairs
        if len(processed_opts) % 2 != 0:
            logger.error(f"Processed options list has an odd number of elements: {processed_opts}. Configuration might be incorrect.")
            # Decide whether to raise an error or proceed cautiously
            # raise ValueError(f"Processed override list has odd length: {processed_opts}")
        else:
            logger.info(f"Merging processed opts: {processed_opts}")
            cfg.merge_from_list(processed_opts)

    cfg.freeze()
    return cfg
# ****** END OF MODIFIED FUNCTION ******


def main(args):
    cfg = setup_cfg(args) # This now calls the modified setup_cfg
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    # Initialize logger *after* output dir might be set by config
    setup_logger(cfg.OUTPUT_DIR)

    # Log effective args and config *after* logger is set up
    logger.info("***** Effective Arguments *****")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        logger.info("{}: {}".format(key, args.__dict__[key]))
    logger.info("***** Effective Config *****")
    logger.info(f"\n{cfg}")


    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # print_args(args, cfg) # Logging above replaces this
    logger.info("Collecting env info ...")
    logger.info("** System info **\n{}\n".format(collect_env_info()))

    # Check if WANDB entity/project are set (e.g., via env vars or config)
    # You might want to make project/entity configurable via cfg too
    wandb_project = "my_fed_project"
    wandb_entity = "mohd-taha82442-iit-bombay"
    run_name = "exp15_my_model_generalization_all_parameters_leanrable_and_pooling_layers_and_depth_3_10_cleint_distribution_depth_6" # Consider making name dynamic or pass via opts

    # Initialize wandb - ensure config passed contains the final merged values
    # Passing dictionary version of cfg makes it easily viewable in W&B UI
    wandb.init(
        project=wandb_project,
        entity=wandb_entity,
        name=run_name,
        config=cfg # Pass the finalized config object (wandb converts it)
    )
    logger.info(f"Wandb run initialized: {wandb.run.get_url()}")

    # Now build trainer with the finalized config
    trainer = build_trainer(cfg)
    logger.info("Trainer built successfully.")

    if args.eval_only:
        logger.info("Starting evaluation only.")
        trainer.load_model(args.model_dir, epoch=args.load_epoch)
        trainer.test_on_all_clients() # fed change 3
        logger.info("Evaluation finished.")
        return

    if not args.no_train:
        logger.info("Starting training.")
        trainer.train()
        logger.info("Training finished.")

    # Ensure wandb finishes syncing
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ... [argparse arguments remain the same] ...
    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line (KEY VALUE pairs or --KEY=VALUE)",
    )

    args = parser.parse_args()

    # Note: Logger setup is moved inside main() after potential config changes
    # But basic logging like this might be useful for very early errors
    print(f"Starting script with provided args: {args}")

    main(args)
    print("Script finished.")
