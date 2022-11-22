#!/bin/bash

DATASET=imagenet1k
# MODEL=RN50x64
# PRETRAIN=openai
MODEL=ViT-B-16
PRETRAIN=openai

FEWSHOT_K=1
LOSS=l2

GRAD_SAMPLER_BATCH_SIZE=64
CALIBRATION_BATCH_SIZE=64

# for calibration
# --create_calibration_loader \
# --calibration_batch_size ${CALIBRATION_BATCH_SIZE} \

# for grad_sampling
# --create_grad_sampler \
# --grad_sampler_batch_size ${GRAD_SAMPLER_BATCH_SIZE} \

# switch to main directory
cd ..

python clip_benchmark/oneshot_sparsification.py \
    --dataset imagenet1k \
    --dataset_root /mnt/data/imagenet \
    \
    --sparseml_recipe_path recipes/one_shot/fast_obc/fast_obc_resnet_sp=0.90_sequential_Ns=1k_B=128_D=1e-5.yaml \
    --loss ${LOSS} \
    \
    --create_grad_sampler \
    --grad_sampler_batch_size ${GRAD_SAMPLER_BATCH_SIZE} \
    \
    --amp \
    \
    --model ${MODEL} \
    --pretrained ${PRETRAIN} \
    \
    --num_workers 8 \
    \
    --fewshot_k ${FEWSHOT_K} \
    \
    --output_root clip_one_shot/fast_obc/sparsity=0.90/loss=${LOSS}/fewshot_k=${FEWSHOT_K} \
    \
    --seed 42 \
    --fix_attention_layer
