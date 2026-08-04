#!/usr/bin/env bash

# Source this file; do not execute it. The aliases remain valid from any
# working directory and forward extra arguments to the preset runner.
_softmimic_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
printf -v _softmimic_runner_q '%q' "$_softmimic_script_dir/run_mujoco.sh"

alias sm-stand="$_softmimic_runner_q stand"
alias sm-walk="$_softmimic_runner_q walk"

unset _softmimic_script_dir _softmimic_runner_q
