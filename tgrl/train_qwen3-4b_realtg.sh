#!/bin/bash
# Log in to Weights & Biases first (`wandb login`) or export WANDB_API_KEY.
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_MODE=online
eval "$(conda shell.bash hook)"
conda activate graphrl

# Run from PATH_TO_VERL/recipe/tgrl (or set TGRL_DIR to it); data and reward paths are absolute from there
TGRL_DIR="${TGRL_DIR:-$(pwd)}"

nvidia-smi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TGRL_DIR/dataset/realtg/train.parquet" \
    data.val_files="$TGRL_DIR/dataset/realtg/val.parquet" \
    data.train_batch_size=32 \
    data.max_prompt_length=8192 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path="$TGRL_DIR/reward_tgrl.py" \
    custom_reward_function.name=compute_score_list \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.resume_mode=disable \
    trainer.project_name='verl_grpo_tgrl' \
    trainer.experiment_name='qwen3_4b_link_prediction_realtg' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=3 $@