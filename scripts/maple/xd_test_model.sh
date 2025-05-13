#!/bin/bash

# Disable wandb to avoid Protobuf version mismatch issues
export WANDB_DISABLED=true
export PYTHONPATH=.:$PYTHONPATH

# custom config
DATA=/home/ashutosh/taha2/
TRAINER=MaPLeFederatedTester

# Command line arguments
DATASET=$1       # Dataset to test on (e.g., RESICS)
SEED=$2          # Seed value
GENERATE_TSNE=${3:-1}  # Whether to generate t-SNE plot (default: yes)

# Hard-coded model path as requested
MODEL_DIR="/home/ashutosh/taha2/federated_multi_modal/artifacts/aggregator_checkpoint:v115"  # Use 'best' since this is a specific checkpoint version

CFG=vit_b16_c2_ep5_batch4_2ctx_cross_datasets
SHOTS=16

# Set up directories
DIR=output/evaluation/test_model/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
TSNE_DIR=${DIR}/tsne_plots

# Create directories if they don't exist
mkdir -p ${DIR}
mkdir -p ${TSNE_DIR}

echo "Testing model from specific checkpoint: ${MODEL_DIR}"
echo "Testing on dataset: ${DATASET}"
echo "Results will be saved to ${DIR}"

# Base command
CMD="python test_model.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/MaPLeFederated/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch 2"

# Add t-SNE options if requested
if [ "$GENERATE_TSNE" -eq 1 ]; then
    TSNE_FILE="${TSNE_DIR}/${DATASET}_tsne.png"
    echo "Will generate t-SNE plot at: ${TSNE_FILE}"
    CMD="${CMD} \
    --generate-tsne-plot \
    --tsne-output-file ${TSNE_FILE} \
    --tsne-perplexity 30 \
    --tsne-n-iter 1000 \
    --tsne-max-samples 2000 \
    --tsne-high-res \
    --tsne-individual-plots \
    --tsne-show-hulls \
    --tsne-annotate-confidence"
fi

# Execute the command
echo "Running: ${CMD}"
eval ${CMD}

echo "Testing complete. Results saved to ${DIR}"
if [ "$GENERATE_TSNE" -eq 1 ]; then
    echo "t-SNE plot saved to ${TSNE_FILE}"
fi 