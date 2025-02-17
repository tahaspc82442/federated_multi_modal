#!/bin/bash

#cd ../..

# custom config
DATA=/raid/biplab/taha
TRAINER=MaPLe

DATASET=$1
SEED=$2

CFG=vit_b16_c2_ep5_batch4_2ctx_cross_datasets
SHOTS=16