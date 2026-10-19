#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

env_name="${1:?usage: bash ./scripts/train_cslerp.sh <2-tower|8-tower>}"

case "$env_name" in
  2-tower|creative-2-task1)
    env_id="creative-2-task1"
    env_episode_length=200
    ;;
  8-tower|creative-8-task1)
    env_id="creative-8-task1"
    env_episode_length=500
    ;;
  *)
    echo "unknown env: $env_name" >&2
    exit 1
    ;;
esac

python cslerp.py --env-id "$env_id" --env-episode-length "$env_episode_length"
