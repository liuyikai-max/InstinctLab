# python scripts/instinct_rl/train.py \
#     --headless \
#     --task=Instinct-Perceptive-Shadowing-G1-v0 \
#     --num_envs=2048 \
#     # --resume --load_run=<RUN_ID> \
#     # --distributed \

python scripts/instinct_rl/train.py \
    --headless \
    --task=Instinct-Perceptive-Shadowing-X2T2D5-v0 \
    --num_envs=1024 \
    # --resume --load_run=<RUN_ID> \
    # --distributed \

# python scripts/instinct_rl/train.py \
#     --headless \
#     --task=Instinct-BeyondMimic-Plane-X2T2D5-v0 \
#     --num_envs=16384 \