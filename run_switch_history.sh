#!/usr/bin/env bash
set -euo pipefail

bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ftd_python="${FTD_PYTHON:-${HOME}/.conda/envs/llm/bin/python3}"
gsm8k_run="$(realpath -m -- "${FTD_GSM8K_RUN:-${HOME}/ftd_acl2027_generation_20261109}")"
dialogsum_run="$(realpath -m -- "${FTD_DIALOGSUM_RUN:-${HOME}/ftd_acl2027_dialogsum_20261109}")"
ftd_output="$(realpath -m -- "${FTD_HISTORY_OUTPUT:-${HOME}/ftd_acl2027_switch_history_20261110}")"
idle_timeout="${FTD_IDLE_TIMEOUT:-28800}"

if [[ ! -x "$ftd_python" ]]; then
    printf 'Original Python was not found at %s. Set FTD_PYTHON to its executable path.\n' "$ftd_python" >&2
    exit 1
fi
for original_run in "$gsm8k_run" "$dialogsum_run"; do
    if [[ "$ftd_output" == "$original_run" || "$ftd_output" == "$original_run/"* || "$original_run" == "$ftd_output/"* ]]; then
        printf 'Choose a new output directory outside the original run: %s\n' "$original_run" >&2
        exit 1
    fi
done
export PYTHONDONTWRITEBYTECODE=1

runner="$bundle_dir/switch_history_depth.py"
"$ftd_python" -u "$runner" check --task gsm8k --run-dir "$gsm8k_run"
"$ftd_python" -u "$runner" check --task dialogsum --run-dir "$dialogsum_run"

mkdir -p -- "$ftd_output"
exec 9>"$ftd_output/.run.lock"
if ! flock -n 9; then
    printf 'A sweep is already active in %s\n' "$ftd_output" >&2
    exit 1
fi
"$ftd_python" -u "$runner" run --task gsm8k --run-dir "$gsm8k_run" --output "$ftd_output" --idle-timeout "$idle_timeout"
"$ftd_python" -u "$runner" run --task dialogsum --run-dir "$dialogsum_run" --output "$ftd_output" --idle-timeout "$idle_timeout"
"$ftd_python" -u "$runner" pack --output "$ftd_output"
