#!/bin/bash

# scripts/fedclip.sh

# Set -e to exit immediately if a command exits with a non-zero status.
# Set -u to treat unset variables as an error.
set -eu

# Custom config (adapt these as needed)
DATA="/raid/biplab/taha/"  # Path to your datasets directory
TRAINER="FedCLIPFederated"       # Your custom trainer name
CONFIG_FILE="fedclip"           # Base name of the trainer config file
OUTPUT_BASE="output"              # Base output directory

# Get dataset and seed from command-line arguments
DATASET="$1"
SEED="$2"

# Construct the output directory.  This is much more descriptive.
# It includes the dataset, trainer, config file, and seed.
DIR="${OUTPUT_BASE}/${DATASET}/${TRAINER}/${CONFIG_FILE}/seed${SEED}"

# Check if the output directory already exists
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}."
else
    echo "Run this job and save the output to ${DIR}"

    # Construct the full paths to the config files.  This is MORE ROBUST.
    TRAINER_CONFIG="configs/trainers/fedclip/${CONFIG_FILE}.yaml"
    DATASET_CONFIG="configs/datasets/${DATASET}.yaml"

    # Check if the config files exist BEFORE running the Python script
    if [ ! -f "$TRAINER_CONFIG" ]; then
        echo "Error: Trainer config file not found: $TRAINER_CONFIG"
        exit 1
    fi
    if [ ! -f "$DATASET_CONFIG" ]; then
        echo "Error: Dataset config file not found: $DATASET_CONFIG"
        exit 1
    fi


    # Run the Python script (using run_fedclip.py)
    python run_fedclip.py \
        --root "${DATA}" \
        --seed "${SEED}" \
        --trainer "${TRAINER}" \
        --dataset-config-file "${DATASET_CONFIG}" \
        --config-file "${TRAINER_CONFIG}" \
        --output-dir "${DIR}" \
        --backbone "ViT-B/32" \
        --run-name "fedclip_run_${DATASET}_seed${SEED}" #Dynamic name

    # You could add error checking here, e.g., check the exit code of
    # the python command and print an error message if it failed.

fi

echo "Script finished."