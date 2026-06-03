# Motion data folder consumed by tasks/shadowing/perceptive/config/x2t2d5/perceptive_shadowing_cfg.py
export MOTION_FOLDER="/home/agiuser/projects/Instinct/InstinctLab/data/perceptive_shadowing_x2/kneelClimbStep1"

python source/instinctlab/instinctlab/tasks/shadowing/play.py \
    --task=Instinct-Perceptive-Shadowing-X2T2D5-Play-v0 \
    --load_run=/home/agiuser/projects/Instinct/InstinctLab/logs/instinct_rl/x2t2d5_perceptive_shadowing/20260508_090122_x2t2d5Perceptive_concatMotionBins__GPU6 \
    --video \
    --video_length=2000 \
    # --exportonnx
