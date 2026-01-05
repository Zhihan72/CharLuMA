#!/bin/bash

PROMPT_VERSION=plain
MODEL_VERSION="deepseek-coder-1.3b-instruct"

deepspeed  --include localhost:0,1,2,3,4,5,6,7 llava/train/train_mem_shareprivate.py \
    --deepspeed ./scripts_imgcoder/zero2.json \
    --model_name_or_path ./checkpoints/$MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path ./training_dataset/chart2Ncode_train_set_warmup.json \
    --image_folder ./images \
    --vision_tower ./checkpoints/siglip-so400m-patch14-384 \
    --tune_mm_mlp_adapter True \
    --pretrain_mm_mlp_adapter ./checkpoints/ckpt_pretrain_ChartMoEAlign_json/mm_projector.bin \
    --mm_projector_type "mlp_murmoe" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/ckpt_finetune_warmup_Chart2NCode \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --mm_projector_lr 2e-4 \
    --learning_rate 2e-4 \
    --weight_decay 0.05 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
