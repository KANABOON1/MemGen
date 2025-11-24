#!/bin/bash

export DEBUG_MODE=true
export LOG_PATH="./debug_log_2b.txt"
export CUDA_VISIBLE_DEVICES=0
export MAIN_PROCESS_PORT=29507
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_DISABLE=1

# options:
# - Qwen/Qwen2.5-1.5B-Instruct
# - HuggingFaceTB/SmolLM3-3B
REASONER_MODEL="Qwen/Qwen2.5-1.5B-Instruct"   
WEAVER_MODEL="Qwen/Qwen2.5-1.5B-Instruct" 
TRIGGER_MODEL="Qwen/Qwen2.5-1.5B-Instruct" 

# Dataset configs
DATASET_NAME="gsm8k"  # options: gsm8k, gpqa, kodcode, triviaqa
DATASET_MODE="grpo"   # options: sft or grpo

# MemGen configs
TRAIN_METHOD="grpo"   # options: sft or grpo

# Augmentation configs:
# - For gsm8k, gpqa, kodcode: MAX_PROMPT_AUG_NUM=1, MAX_INFERENCE_AUG_NUM=5
# - For triviaqa:             MAX_PROMPT_AUG_NUM=6, MAX_INFERENCE_AUG_NUM=0
MAX_PROMPT_AUG_NUM=1
MAX_INFERENCE_AUG_NUM=5
PROMPT_LATENTS_LEN=16
INFERENCE_LATENTS_LEN=8


LOAD_WEAVER_PATH=null

# train
python -m accelerate.commands.launch \
    --config_file=configs/zero2.yaml \
    main.py \
    --cfg-path configs/latent_memory/${DATASET_NAME}.yaml \
    --options \
    model.model_name ${REASONER_MODEL} \
    model.load_model_path ${LOAD_WEAVER_PATH} \
    model.max_prompt_aug_num ${MAX_PROMPT_AUG_NUM} \
    model.max_inference_aug_num ${MAX_INFERENCE_AUG_NUM} \
    model.weaver.model_name ${WEAVER_MODEL} \
    model.weaver.prompt_latents_len ${PROMPT_LATENTS_LEN} \
    model.weaver.inference_latents_len ${INFERENCE_LATENTS_LEN} \
    model.trigger.model_name ${TRIGGER_MODEL} \
    model.trigger.active True \
    datasets.mode ${DATASET_MODE} \
    run.mode train \
    run.train_weaver False \
    run.train_trigger True \
    run.train_trigger_method ${TRAIN_METHOD} \
    run.trigger.grpo.per_device_train_batch_size 8 \
    run.trigger.grpo.per_device_eval_batch_size 8 \
    run.trigger.grpo.num_train_epochs 1 \
    run.trigger.grpo.num_generations 8 \
    run.trigger.grpo.gradient_accumulation_steps 4 \
    run.interaction.do_sample True \
    run.interaction.temperature 1.0 \
    run.interaction.max_response_length 1024 \





