#!/bin/bash

DATASET=imagenet1k
MODEL=ViT-B-16
PRETRAIN=openai

FEWSHOT_K=16
LOSS=l2
BATCH_SIZE=128

export WANDB_ENTITY=oViT
export WANDB_PROJECT=SparseCLIP
export WANDB_NAME=${MODEL}_${PRETRAIN}_one-shot_pruning

cd ..

python clip_benchmark/cli.py \
    --dataset imagenet1k \
    --dataset_root /mnt/data/imagenet \
    --checkpoint_path clip_one_shot/fast_obc/sparsity=0.5/loss=l2/fewshot_k=1/ViT-B-16-openai-imagenet1k/checkpoint.pth \
    \
    --model ${MODEL} \
    --pretrained ${PRETRAIN} \
    --amp \
    \
    --num_workers 16 \
    --batch_size ${BATCH_SIZE} \
    \
    --task zeroshot_classification \
    \
    --output results/zero_shot_classification/ovit/sparsity=0.50/loss=${LOSS}/fewshot_k=${FEWSHOT_K}/${MODEL}/${PRETRAIN}/result.json \
    \
    --seed 42 \
    --fix_attention_layer \
    \
    --log_wandb \
 