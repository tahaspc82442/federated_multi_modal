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
    print("Generating t-SNE plot...")

    model_for_embeddings = None
    data_loader = None
    class_names_map = None # For label to class name mapping if needed for legend
    num_distinct_classes = 0
    model_dtype = torch.float16 # Default, will try to get from model

    if hasattr(trainer, 'clients') and len(trainer.clients) > 0:
        # For federated trainers (MaPLeFederated and MaPLeFederatedTester)
        # The model is in the clients list
        print("Using client 0's model for embedding extraction")
        client_trainer = trainer.clients[0]
        model_for_embeddings = client_trainer.model
        data_loader = trainer.client_data_managers[0].test_loader
        # Federated trainers store global label id to classname in lab2cname
        class_names_map = trainer.lab2cname
        num_distinct_classes = len(trainer.lab2cname)
        if hasattr(model_for_embeddings, 'dtype'):
            model_dtype = model_for_embeddings.dtype
    elif hasattr(trainer, 'model'):
        # For non-federated trainers
        model_for_embeddings = trainer.model
        data_loader = trainer.test_loader
        if trainer.dm.dataset.classnames:
            class_names_map = {i: name for i, name in enumerate(trainer.dm.dataset.classnames)}
            num_distinct_classes = trainer.dm.dataset.num_classes
        else: # Fallback if classnames are not directly available
            # This part might need adjustment based on how num_classes is determined for unknown datasets
            num_distinct_classes = cfg.MODEL.NUM_CLASSES

        if hasattr(model_for_embeddings, 'dtype'):
            model_dtype = model_for_embeddings.dtype
    else:
        print("ERROR: Could not determine model structure. Trainer has no 'model' or 'clients' attribute.")
        return

    if model_for_embeddings is None or data_loader is None:
        print("Could not determine model or data loader for t-SNE. Skipping.")
        return

    # Ensure the model used for embeddings is the underlying module if wrapped (e.g., by DataParallel)
    if isinstance(model_for_embeddings, torch.nn.DataParallel):
        model_for_embeddings = model_for_embeddings.module

    # Set model to evaluation mode
    model_for_embeddings.eval()
    
    # Determine if this is a MaPLe or CustomCLIP model (needs special handling)
    is_maple_model = 'prompt_learner' in dir(model_for_embeddings)
    
    all_features = []
    all_labels = []

    print(f"Extracting features using device: {trainer.device}")
    print(f"Using {'MaPLe/CustomCLIP' if is_maple_model else 'standard'} feature extraction method")

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            images = batch["img"].to(trainer.device)
            labels = batch["label"].cpu().numpy()

            # Extract features based on model type
            if is_maple_model:
                # For MaPLe models, we need to handle the full forward pass differently
                # to get image features, because direct encoder access requires additional args
                try:
                    result = model_for_embeddings(images.type(model_dtype), return_feature=True)
                    if isinstance(result, tuple) and len(result) == 2:
                        logits, features = result
                    else:
                        print(f"Warning: Expected tuple of (logits, features) but got {type(result)}. Trying alternative approaches.")
                        # Fallback: try direct encoder access
                        if hasattr(model_for_embeddings, 'image_encoder'):
                            print("Falling back to direct image_encoder access")
                            image_encoder = model_for_embeddings.image_encoder
                            # Get the prompt components needed for the encoder
                            prompts, shared_ctx, deep_compound_prompts_text, deep_compound_prompts_vision = model_for_embeddings.prompt_learner()
                            features = image_encoder(
                                images.type(model_dtype),
                                shared_ctx.to(dtype=model_dtype),
                                [v.to(dtype=model_dtype) for v in deep_compound_prompts_vision],
                                None  # No caption embedding 
                            )
                        else:
                            raise ValueError("Could not extract features from model")
                except Exception as e:
                    print(f"Error during feature extraction with MaPLe model: {e}")
                    raise
            else:
                # For simpler models, direct feature extraction may work
                image_encoder = None
                if hasattr(model_for_embeddings, 'image_encoder'):
                    image_encoder = model_for_embeddings.image_encoder
                elif hasattr(model_for_embeddings, 'visual'):  # Common in CLIP models
                    image_encoder = model_for_embeddings.visual
                else:
                    print("Could not find image_encoder/visual attribute on the model. Skipping t-SNE.")
                    return
                
                image_encoder.eval()
                features = image_encoder(images.type(model_dtype))
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels)
            
            if sum(len(f) for f in all_features) >= args.tsne_max_samples:
                print(f"Reached max samples for t-SNE ({args.tsne_max_samples}).")
                break
        
    if not all_features:
        print("No features extracted. Skipping t-SNE plot.")
        return

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Determine number of unique classes for coloring and markers
    unique_class_labels = np.unique(all_labels)
    
    # Subsample if we exceeded max_samples slightly due to batching
    if len(all_features) > args.tsne_max_samples:
        indices = np.random.choice(len(all_features), args.tsne_max_samples, replace=False)
        all_features = all_features[indices]
        all_labels = all_labels[indices]
        # Recalculate unique labels as some might have been removed during sampling
        unique_class_labels = np.unique(all_labels)

    print(f"Running t-SNE on {len(all_features)} samples...")
    tsne = TSNE(
        n_components=2,
        perplexity=args.tsne_perplexity,
        max_iter=args.tsne_n_iter,  # Changed from n_iter to max_iter
        random_state=cfg.SEED if cfg.SEED >=0 else None,
        init='pca', # PCA initialization is generally good
        learning_rate='auto'
    )
    tsne_results = tsne.fit_transform(all_features)

    # Create the figure and axes before computing metrics and other plot elements
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111)
    
    # Calculate metrics
    silhouette_avg = "N/A"
    dbi_score = "N/A"
    ch_score = "N/A"
    
    unique_labels_count = len(unique_class_labels)
    if unique_labels_count > 1 and unique_labels_count < len(all_labels):
        try:
            silhouette_avg = f"{silhouette_score(all_features, all_labels):.4f}"
        except ValueError as e:
            print(f"Could not calculate Silhouette Score: {e}")
        try:
            dbi_score = f"{davies_bouldin_score(all_features, all_labels):.4f}"
        except ValueError as e:
            print(f"Could not calculate Davies-Bouldin Index: {e}")
        try:
            ch_score = f"{calinski_harabasz_score(all_features, all_labels):.0f}" # Example image format
        except ValueError as e:
            print(f"Could not calculate Calinski-Harabasz Index: {e}")
    else:
        print("Not enough distinct labels or samples to calculate clustering metrics.")

    # Calculate per-class silhouette scores if possible
    class_silhouette_scores = {}
    try:
        if silhouette_avg != "N/A" and unique_labels_count > 1:
            from sklearn.metrics import silhouette_samples
            silhouette_vals = silhouette_samples(all_features, all_labels)
            for label_val in unique_class_labels:
                indices = all_labels == label_val
                if np.sum(indices) > 0:
                    class_silhouette_scores[label_val] = np.mean(silhouette_vals[indices])
    except Exception as e:
        print(f"Could not calculate per-class silhouette scores: {e}")
        class_silhouette_scores = {}  # Reset to empty in case of any error

    # If class_names_map is available and num_distinct_classes was derived from it, use its length.
    # Otherwise, use the count of unique labels found in the current data.
    effective_num_classes = num_distinct_classes if num_distinct_classes > 0 else len(unique_class_labels)
    
    # Generate distinct colors - improve with more distinct colormap for better differentiation
    if effective_num_classes <= 10: # Max Tableau
        plot_colors = list(mcolors.TABLEAU_COLORS.values())[:effective_num_classes]
    elif effective_num_classes <= 20: # Max tab20
        cmap = plt.get_cmap('tab20')
        plot_colors = [cmap(i) for i in np.linspace(0, 1, effective_num_classes)]
    else: # For more than 20, use hsv which cycles through hues
        cmap = plt.get_cmap('hsv')
        plot_colors = [cmap(i) for i in np.linspace(0, 0.9, effective_num_classes)]  # 0.9 to avoid red-red overlap

    # Markers
    markers = ['o', 'v', '^', '<', '>', 's', 'p', '*', '+', 'X', 'D', 'd', '|', '_'] # 14 markers
    
    # Map labels to colors and markers
    # Ensure we handle cases where labels might not be 0-indexed contiguous
    try:
        label_to_idx = {label: i for i, label in enumerate(unique_class_labels)}
    except Exception as e:
        print(f"Error creating label_to_idx mapping: {e}")
        label_to_idx = {} # Fallback empty dictionary
    
    # For legend: track scatter plot objects and their class names
    legend_handles = []
    legend_labels = []

    # Plotting code wrapped in try-except
    try:
        # Plot each class
        for label_val in unique_class_labels:
            try:
                idx_for_style = label_to_idx.get(label_val, 0)  # Use get with default 0
                indices = all_labels == label_val
                
                # Skip classes with no samples
                if not np.any(indices):
                    continue
                
                # Get class name from mapping if available
                if class_names_map and int(label_val) in class_names_map:
                    class_name = class_names_map[int(label_val)]
                    # Format class name for better readability
                    class_name = class_name.replace('_', ' ').title()
                else:
                    class_name = f"Class {label_val}"
                
                # Plot this class
                scatter = ax.scatter(
                    tsne_results[indices, 0],
                    tsne_results[indices, 1],
                    color=plot_colors[idx_for_style % len(plot_colors)],
                    marker=markers[idx_for_style % len(markers)],
                    alpha=0.7,
                    s=70,  # Slightly larger points
                    edgecolors='w',  # White edge for better visibility
                    linewidths=0.5
                )
                
                # Add to legend only if this class has a reasonable number of samples
                # (prevents legend from becoming too large)
                num_samples = np.sum(indices)
                if num_samples > max(5, len(all_labels) * 0.01):  # At least 5 samples or 1% of total
                    legend_handles.append(scatter)
                    legend_labels.append(f"{class_name} ({num_samples})")
            except Exception as e:
                print(f"Error plotting class {label_val}: {e}")
                continue  # Skip this class
        
        # Add convex hulls around clusters with sufficient points
        try:
            if hasattr(args, 'tsne_show_hulls') and args.tsne_show_hulls:
                from scipy.spatial import ConvexHull
                
                for label_val in unique_class_labels:
                    indices = all_labels == label_val
                    points = tsne_results[indices]
                    
                    # Need at least 4 points for convex hull
                    if len(points) >= 4:
                        try:
                            hull = ConvexHull(points)
                            hull_color = plot_colors[label_to_idx.get(label_val, 0) % len(plot_colors)]
                            
                            # Draw hull with some transparency
                            for simplex in hull.simplices:
                                ax.plot(points[simplex, 0], points[simplex, 1], 
                                        color=hull_color, alpha=0.5, linestyle='-', linewidth=1.5)
                        except Exception as e:
                            print(f"Could not calculate convex hull for class {label_val}: {e}")
        except Exception as e:
            print(f"Error in convex hull calculation: {e}")
        
        # Get dataset name for title
        dataset_name = cfg.DATASET.NAME if hasattr(cfg.DATASET, 'NAME') else 'Unknown Dataset'

        # Set title
        title_str = (
            f"t-SNE Plot of Feature Embeddings: {dataset_name}\n"
            f"Silhouette: {silhouette_avg} | DBI: {dbi_score} | CH Index: {ch_score}"
        )
        ax.set_title(title_str, fontsize=14)
        
        # Add "Our model" text below the plot
        plt.figtext(0.5, 0.02, "Our Model", ha="center", fontsize=24, color='gray')
        
        # Add confidence annotations for classes with good separation
        try:
            if class_silhouette_scores and hasattr(args, 'tsne_annotate_confidence') and args.tsne_annotate_confidence:
                # Sort classes by silhouette score
                sorted_scores = sorted(class_silhouette_scores.items(), key=lambda x: x[1], reverse=True)
                
                # Get top and bottom classes (if available)
                top_classes = sorted_scores[:min(3, len(sorted_scores))]
                bottom_classes = sorted_scores[-min(2, len(sorted_scores)):] if len(sorted_scores) > 2 else []
                
                if top_classes:
                    # Add top class annotations
                    plt.figtext(0.02, 0.98, "Best separated classes:", ha="left", va="top", 
                            fontsize=10, fontweight='bold', color='darkgreen')
                    
                    for i, (label_val, score) in enumerate(top_classes):
                        try:
                            if class_names_map and int(label_val) in class_names_map:
                                class_name = class_names_map[int(label_val)].replace('_', ' ').title()
                            else:
                                class_name = f"Class {label_val}"
                                
                            plt.figtext(0.02, 0.95 - i*0.03, f"{class_name}: {score:.3f}", 
                                    ha="left", va="top", fontsize=9, 
                                    color=plot_colors[label_to_idx.get(label_val, 0) % len(plot_colors)])
                        except Exception as e:
                            print(f"Error annotating top class {label_val}: {e}")
                
                if bottom_classes:
                    # Add bottom class annotations
                    plt.figtext(0.02, 0.85, "Least separated classes:", ha="left", va="top", 
                            fontsize=10, fontweight='bold', color='darkred')
                    
                    for i, (label_val, score) in enumerate(bottom_classes):
                        try:
                            if class_names_map and int(label_val) in class_names_map:
                                class_name = class_names_map[int(label_val)].replace('_', ' ').title()
                            else:
                                class_name = f"Class {label_val}"
                                
                            plt.figtext(0.02, 0.82 - i*0.03, f"{class_name}: {score:.3f}", 
                                    ha="left", va="top", fontsize=9, 
                                    color=plot_colors[label_to_idx.get(label_val, 0) % len(plot_colors)])
                        except Exception as e:
                            print(f"Error annotating bottom class {label_val}: {e}")
        except Exception as e:
            print(f"Error adding confidence annotations: {e}")

        # Remove ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add legend with reasonable size and placement
        if legend_handles:
            # Determine optimal number of columns based on number of classes
            num_classes_in_legend = len(legend_handles)
            if num_classes_in_legend <= 10:
                ncol = 1
            elif num_classes_in_legend <= 20:
                ncol = 2
            else:
                ncol = 3
                
            # Place legend to the right of the plot
            legend = ax.legend(
                legend_handles, 
                legend_labels,
                loc='center left', 
                bbox_to_anchor=(1, 0.5),
                frameon=True,
                fontsize=10,
                ncol=ncol,
                title="Classes",
                title_fontsize=12
            )
            legend.get_frame().set_alpha(0.8)  # Semi-transparent legend background
            
        # Adjust layout to make room for legend
        plt.tight_layout()
    except Exception as e:
        print(f"Error in main plotting routine: {e}")
        # If plotting fails, try to create a very simple plot as a fallback
        try:
            plt.clf()  # Clear the figure
            ax = plt.subplot(111)
            ax.text(0.5, 0.5, f"Error generating t-SNE plot: {str(e)}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        except Exception as nested_e:
            print(f"Even fallback plotting failed: {nested_e}")
            # At this point, we just let it fail

    # Ensure output directory exists
    output_dir = os.path.dirname(args.tsne_output_file)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"Saving t-SNE plot to: {args.tsne_output_file}")
    plt.savefig(args.tsne_output_file, bbox_inches='tight', dpi=150)
    print(f"t-SNE plot saved successfully to: {args.tsne_output_file}")
    plt.close(fig)

    # Save individual class plots if requested
    try:
        if hasattr(args, 'tsne_individual_plots') and args.tsne_individual_plots:
            individual_dir = os.path.join(os.path.dirname(args.tsne_output_file), "class_plots")
            os.makedirs(individual_dir, exist_ok=True)
            print(f"Saving individual class plots to {individual_dir}")
            
            # Reuse the t-SNE results to create individual class plots
            for label_val in unique_class_labels:
                try:
                    idx_for_style = label_to_idx[label_val]
                    indices = all_labels == label_val
                    
                    # Skip classes with too few samples
                    num_samples = np.sum(indices)
                    if num_samples < 5:
                        continue
                    
                    # Get class name
                    if class_names_map and int(label_val) in class_names_map:
                        class_name = class_names_map[int(label_val)]
                        # Format class name for better readability
                        class_name_formatted = class_name.replace('_', ' ').title()
                        # For filename, use original with underscores
                        filename_class = class_name.replace(' ', '_').lower()
                    else:
                        class_name_formatted = f"Class {label_val}"
                        filename_class = f"class_{label_val}"
                    
                    # Create a small figure with just this class
                    fig_class, ax_class = plt.subplots(figsize=(8, 6))
                    
                    # Plot all data points as faded gray background
                    ax_class.scatter(
                        tsne_results[:, 0], 
                        tsne_results[:, 1],
                        color='lightgray', 
                        alpha=0.2, 
                        s=30
                    )
                    
                    # Highlight this class
                    ax_class.scatter(
                        tsne_results[indices, 0],
                        tsne_results[indices, 1],
                        color=plot_colors[idx_for_style % len(plot_colors)],
                        marker=markers[idx_for_style % len(markers)],
                        alpha=0.9,
                        s=100,
                        edgecolors='white',
                        linewidths=0.7,
                        label=class_name_formatted
                    )
                    
                    ax_class.set_title(f"t-SNE: {class_name_formatted} ({num_samples} samples)", fontsize=14)
                    ax_class.legend(loc='best')
                    ax_class.set_xticks([])
                    ax_class.set_yticks([])
                    
                    # Save this class plot
                    class_output_file = os.path.join(individual_dir, f"{filename_class}.png")
                    plt.tight_layout()
                    plt.savefig(class_output_file, bbox_inches='tight', dpi=150)
                    plt.close(fig_class)
                except Exception as e:
                    print(f"Error creating individual plot for class {label_val}: {e}")
    except Exception as e:
        print(f"Error generating individual class plots: {e}")


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
        
        # If high resolution requested, save that too
        if args.tsne_high_res:
            high_res_path = args.tsne_output_file.replace('.png', '_high_res.png')
            print(f"Saving high-resolution version to: {high_res_path}")
            # Increase DPI for high resolution
            plt.savefig(high_res_path, bbox_inches='tight', dpi=300)
            print(f"High-resolution t-SNE plot saved to: {high_res_path}")

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