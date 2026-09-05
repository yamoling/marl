#!/usr/bin/env bash
# Driver for the ACER validation experiments (see scripts/train_acer_gym.py and
# scripts/train_acer_lle.py). Runs unattended:
#   1. waits for the single-agent CartPole studies to finish;
#   2. restores the legacy-gpu PyTorch so the GTX 1080 Ti GPUs are usable;
#   3. plots the CartPole replay-ratio and ablation studies;
#   4. trains the multi-agent LLE study on the GPUs;
#   5. plots the LLE study.
# Every stage is banner-logged, so `tail -f acer_validation.log` shows where it is.
# Never call plain `uv run` here: it reinstalls a PyTorch that has no sm_61 kernels.
# Usage: run_acer_validation.sh [all|lle]   ("lle" resumes at the multi-agent stage)
set -u
cd /home/yamoling/marl

STAGE="${1:-all}"
UV="uv run --extra=legacy-gpu"
banner() { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }

if [ "$STAGE" = "all" ]; then

banner "Waiting for the CartPole studies to finish"
while pgrep -f "[t]rain_acer_gym.py" >/dev/null 2>&1; do sleep 60; done
echo "CartPole studies done."

banner "Restoring the legacy-gpu environment (torch < 2.8)"
# `uv sync --extra=legacy-gpu` alone leaves torch 2.7.1 without its cudnn libraries: the two-step
# upgrade is the recipe documented in .agents/handoffs/2026-09-04-laies-paper-reproduction.md.
uv sync -U && uv sync -U --extra=legacy-gpu

banner "Plotting the CartPole replay-ratio study"
$UV python scripts/analyse_acer.py --logdir logs/acer-replay/CartPole-v1 \
    --output .agents/reports/acer-replay-cartpole.png \
    --title "ACER on CartPole-v1: replay ratio x trust region (5 seeds)"

banner "Plotting the CartPole ablation study"
$UV python scripts/analyse_acer.py --logdir logs/acer-ablation/CartPole-v1 \
    --output .agents/reports/acer-ablation-cartpole.png \
    --title "ACER on CartPole-v1: component ablation (5 seeds)"

fi  # end of the single-agent stage

banner "Checking that the GPUs are usable"
$UV python -c "
import torch
assert torch.cuda.is_available(), 'no CUDA device'
x = torch.randn(512, 512, device='cuda:0')
print('torch', torch.__version__, 'on', torch.cuda.device_count(), 'GPUs; cuda:0 matmul ok')
"
if [ $? -ne 0 ]; then
    banner "ABORTING: the GPUs are not usable, not launching the LLE study"
    echo "Recovery: uv sync -U && uv sync -U --extra=legacy-gpu"
    exit 1
fi

banner "Training the multi-agent LLE study (6 conditions x 4 seeds x 500k steps)"
if [ -d logs/acer-lle3 ]; then
    mv logs/acer-lle3 "logs/acer-lle3.$(date +%Y%m%d-%H%M%S)"
fi
$UV python scripts/train_acer_lle.py \
    --level 3 --obs-type layered --n-steps 500000 --n-seeds 4 \
    --n-jobs 12 --device round-robin --test-interval 10000 --n-tests 10

banner "Plotting the LLE study"
$UV python scripts/analyse_acer.py --logdir logs/acer-lle3 \
    --metric score-0 --output .agents/reports/acer-lle3-score.png \
    --title "ACER vs PPO on LLE level 3 (4 seeds)" || true
$UV python scripts/analyse_acer.py --logdir logs/acer-lle3 \
    --metric exit_rate --output .agents/reports/acer-lle3-exit-rate.png \
    --title "ACER vs PPO on LLE level 3: exit rate (4 seeds)" || true

banner "ALL DONE"
