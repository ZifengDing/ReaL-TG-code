#!/bin/bash
# Log in to Weights & Biases first (`wandb login`) or export WANDB_API_KEY.
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_MODE=online
eval "$(conda shell.bash hook)"
conda activate graphrl

data_path=dataset/realtg/test.parquet
save_path=logs/test_generation/realtg/test_realtg_4b.parquet
model_path=merged_checkpoints/verl_grpo_tgrl/realtg_4b

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=4 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$save_path \
    model.path=$model_path \
    +model.trust_remote_code=True \
    rollout.temperature=0.6 \
    rollout.top_k=20 \
    rollout.top_p=0.95 \
    rollout.prompt_length=8192 \
    rollout.response_length=16384 \
    rollout.tensor_model_parallel_size=4 \
    rollout.gpu_memory_utilization=0.8 \
    rollout.enable_chunked_prefill=False \
