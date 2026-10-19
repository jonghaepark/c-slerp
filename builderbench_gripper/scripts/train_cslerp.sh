#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

env_name="${1:?usage: bash ./scripts/train_cslerp.sh <3-tower|4-tower|4-pyramid|5-tower>}"

case "$env_name" in
  3-tower|gripper-creative-3-task1)
    env_id="gripper-creative-3-task1"
    num_eval_steps=400
    num_reset_steps=400
    num_timesteps=1000000000
    ;;
  4-tower|gripper-creative-4-task1)
    env_id="gripper-creative-4-task1"
    num_eval_steps=800
    num_reset_steps=800
    num_timesteps=2000000000
    ;;
  4-pyramid|gripper-creative-4-task2)
    env_id="gripper-creative-4-task2"
    num_eval_steps=800
    num_reset_steps=800
    num_timesteps=2000000000
    ;;
  5-tower|gripper-creative-5-task1)
    env_id="gripper-creative-5-task1"
    num_eval_steps=1600
    num_reset_steps=1600
    num_timesteps=4000000000
    ;;
  *)
    echo "unknown env: $env_name" >&2
    exit 1
    ;;
esac

args=(--env-id "$env_id")
if [[ -n "${num_timesteps:-}" ]]; then
  args+=(
    --num-eval-steps "$num_eval_steps"
    --num-reset-steps "$num_reset_steps"
    --num-timesteps "$num_timesteps"
  )
fi

python cslerp.py "${args[@]}"
