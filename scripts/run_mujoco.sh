#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: run_mujoco.sh PRESET [DEPLOYMENT_OPTIONS]

Presets:
  stand  StaticStand-SoftMimic with stand.csv (no walking commands)
  walk   GMTWalkStand-SoftMimic with walk.csv (recorded direction/steps)

Examples:
  ./scripts/run_mujoco.sh stand
  ./scripts/run_mujoco.sh walk --time-limit 20
EOF
}

if [[ $# -eq 0 || "$1" == "-h" || "$1" == "--help" ]]; then
    usage
    exit 0
fi

preset="$1"
shift

case "$preset" in
    stand)
        policy="../../pretrained_models/2025-09-26_03-54-58_StaticStand-SoftMimic/model_48000.jit"
        motion="stand.csv"
        ;;
    walk)
        policy="../../pretrained_models/2025-09-26_03-57-30_GMTWalkStand-SoftMimic/model_42000.jit"
        motion="walk.csv"
        ;;
    *)
        printf 'Unknown preset: %s\n\n' "$preset" >&2
        usage >&2
        exit 2
        ;;
esac

cd "$repo_root"
exec uv run python softmimic_deploy/src/deploy_policy_interface.py \
    --interface mujoco \
    --policy "$policy" \
    --motion_path "$motion" \
    --render \
    "$@"
