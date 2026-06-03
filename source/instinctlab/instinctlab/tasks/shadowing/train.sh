# Motion data folder consumed by tasks/shadowing/perceptive/config/x2t2d5/perceptive_shadowing_cfg.py
export MOTION_FOLDER="/home/agiuser/projects/Instinct/InstinctLab/data/perceptive_shadowing_x2/kneelClimbStep1"

python scripts/instinct_rl/train.py \
    --headless \
    --task=Instinct-Perceptive-Shadowing-X2T2D5-v0 \
    --num_envs=1024 \
    # --resume --load_run=<RUN_ID> \
    # --distributed \
