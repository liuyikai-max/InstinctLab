CUDA_VISIBLE_DEVICES=0 python scripts/instinct_rl/train.py \
    --headless \
    --task=Instinct-Perceptive-Shadowing-G1-v0 \
    --num_envs=2048 \
    # --resume --load_run=<RUN_ID> \
    # --distributed \