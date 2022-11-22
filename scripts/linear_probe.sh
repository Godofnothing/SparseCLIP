#!/bin/bash

DATASET=imagenet1k
MODEL=ViT-g-14
PRETRAIN=laion2b_s12b_b42k

FEWSHOT_K=1
LOSS=l2
BATCH_SIZE=128

export WANDB_ENTITY=oViT
export WANDB_PROJECT=SparseCLIP
export WANDB_NAME=linear_probe_${MODEL}_${PRETRAIN}

# --checkpoint_path clip_one_shot/fast_obc/sparsity=0.5/loss=kl_div/fewshot_k=1/ViT-L-14-laion400m_e32-imagenet1k/checkpoint.pth \
cd ..

python clip_benchmark/cli.py \
    --dataset imagenet1k \
    --dataset_root /mnt/data/imagenet \
    \
    --model ${MODEL} \
    --pretrained ${PRETRAIN} \
    --amp \
    \
    --num_workers 16 \
    --batch_size ${BATCH_SIZE} \
    \
    --task linear_probe \
    --fewshot_epochs 10 \
    --fewshot_lr 0.1 \
    \
    --output results/linear_probe/dense/fewshot_k=1/${MODEL}/${PRETRAIN}/result.json \
    \
    --seed 42 \
    --fix_attention_layer \
    \
    --feature_root features/dense/fewshot_k=1/${MODEL}/${PRETRAIN} \
    \
    --log_wandb \
    --log_interval 100 \
 