#!/bin/bash

# Default environment
ENV_NAME=${1:-"sokoban"}

# Set environment variables
export PYTHONHASHSEED=10000
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_VISIBLE_DEVICES=0
export RAY_ADDRESS='127.0.0.1:6379'

# Execute the training command directly
python -m verl.trainer.main_mppo \
    multi_processing=ray \
    +ray.address=141.142.254.51:6379 \
    +ray.redis_password=5241590000000000 \
    data.train_files=data/sokoban/train.parquet \
    data.val_files=data/sokoban/test.parquet \
    data.train_data_num=null \
    data.val_data_num=50 \
    data.train_batch_size=4 \
    data.val_batch_size=10 \
    data.max_prompt_length=1800 \
    data.max_response_length=400 \
    data.max_start_length=400 \
    data.max_obs_length=200 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=False \
    actor_rollout_ref.rollout.n_agent=4 \
    actor_rollout_ref.rollout.temperature=0.1 \
    actor_rollout_ref.actor.state_masking=False \
    trainer.logger=['wandb'] \
    +trainer.val_only=false \
    +trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=200 \
    trainer.project_name=RAGEN \
    trainer.experiment_name=new_test \
    trainer.total_epochs=5 \
    trainer.total_training_steps=null \
    +trainer.ref_update_steps=null \
    env.name=sokoban_dual \
    +env.dim_x=6 \
    +env.dim_y=6 \
    +env.num_boxes=1 \
    +env.max_steps=100 \
    +env.search_depth=30 \
    +env.dual_agent=True \
    max_turns=3 \
    logging.log_images=true \
    logging.log_image_dir=log/trajectory \
    logging.log_image_step_size=4 \
    logging.log_n_image_per_batch=32 \
    2>&1 | tee debug.log