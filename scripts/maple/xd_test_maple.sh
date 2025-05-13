#!/bin/bash

#cd ../..

# custom config
DATA=/raid/biplab/taha
TRAINER=MaPLe

DATASET=$1
SEED=$2
TRAINEDON=$3
EP=$4

CFG=vit_b16_c2_ep5_batch4_2ctx_cross_datasets
SHOTS=16

#DIR=output/evaluation/${TRAINER}/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
DIR=output/evaluation/MaPLeFederated/${CFG}_${SHOTS}shots/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Skip this job"
else
    echo "Run this job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir output/${TRAINEDON}/MaPLeFederated/${CFG}_${SHOTS}shots/seed${SEED} \
    --load-epoch ${EP} \
    --eval-only
    
    # Generate t-SNE visualization with the pre-trained model
    TSNE_DIR=${DIR}/tsne_plots
    mkdir -p ${TSNE_DIR}
    
    echo "Generating t-SNE plots in ${TSNE_DIR}"
    
    python test_model.py \
    --model-dir output/${TRAINEDON}/MaPLeFederated/${CFG}_${SHOTS}shots/seed${SEED} \
    --load-epoch ${EP} \
    --trainer MaPLeFederatedTester \
    --config-file configs/trainers/MaPLeFed/${DATASET}.yaml \
    --generate-tsne-plot \
    --tsne-output-file ${TSNE_DIR}/${DATASET}_tsne.png \
    --tsne-perplexity 40.0 \
    --tsne-max-samples 3000 \
    --tsne-high-res \
    --tsne-individual-plots \
    --tsne-show-hulls \
    --tsne-annotate-confidence \
    DATASET.NAME ${DATASET}
fi