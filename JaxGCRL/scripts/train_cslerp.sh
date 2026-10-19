#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

env_name="${1:?usage: bash ./scripts/train_cslerp.sh <env>}"

case "$env_name" in
  ant_big_maze|ant_u4_maze)
    eval_env="${env_name}_eval"
    total_env_steps=200000000
    episode_length=1001
    ;;
  ant_u5_maze)
    eval_env="${env_name}_eval"
    total_env_steps=400000000
    episode_length=1001
    ;;
  ant_extreme_maze)
    eval_env="${env_name}_eval"
    total_env_steps=400000000
    episode_length=2001
    max_waypoint_depth=10
    ;;
  humanoid)
    eval_env=""
    total_env_steps=200000000
    episode_length=1001
    ;;
  humanoid_u_maze)
    eval_env=""
    total_env_steps=400000000
    episode_length=1001
    ;;
  *)
    echo "unknown env: $env_name" >&2
    exit 1
    ;;
esac

args=(--env "$env_name" --total-env-steps "$total_env_steps" --episode-length "$episode_length")
if [[ -n "$eval_env" ]]; then
  args+=(--eval-env "$eval_env")
fi
if [[ -n "${max_waypoint_depth:-}" ]]; then
  args+=(--max-waypoint-depth "$max_waypoint_depth")
fi

python run.py "${args[@]}"
