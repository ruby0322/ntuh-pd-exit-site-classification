#!/usr/bin/env bash
# Extra runs after run_sweep.sh (wide backbone refinements)
set -uo pipefail
cd "$(dirname "$0")/.."
COMMIT=$(git rev-parse --short HEAD)
RESULTS=results.tsv
mkdir -p experiments

append_row() {
  local status=$1 desc=$2 log=$3
  if [[ "$status" == crash ]]; then
    printf '%s\t0\t0\t0.0\tcrash\t%s\n' "$COMMIT" "$desc" >> "$RESULTS"
    return
  fi
  local mc bin mem_mb mem_gb
  mc=$(grep '^mc_acc:' "$log" | tail -1 | awk '{print $2}')
  bin=$(grep '^bin_acc:' "$log" | tail -1 | awk '{print $2}')
  mem_mb=$(grep '^peak_vram_mb:' "$log" | tail -1 | awk '{print $2}')
  mem_gb=$(python3 -c "print(round(float('${mem_mb:-0}')/1024, 1))")
  printf '%s\t%s\t%s\t%s\tkeep\t%s\n' "$COMMIT" "${mc:-0}" "${bin:-0}" "$mem_gb" "$desc" >> "$RESULTS"
}

run_exp() {
  local name=$1
  shift
  local log="experiments/${name}.log"
  echo "========== $name ==========" | tee "$log"
  if python3 train.py --seed 42 "$@" 2>&1 | tee -a "$log"; then
    append_row keep "$name" "$log"
  else
    append_row crash "$name" "$log"
  fi
  echo "" | tee -a "$log"
}

COMMON=(--epochs 999 --max-train-seconds 300 --log-every 10)
run_exp "e07_wide_adamw" "${COMMON[@]}" --arch wide --optimizer adamw --lr 0.0003 --model-out experiments/model_e07.pt
run_exp "e08_wide_sgd_lr005" "${COMMON[@]}" --arch wide --optimizer sgd --lr 0.005 --model-out experiments/model_e08.pt
cat "$RESULTS"
