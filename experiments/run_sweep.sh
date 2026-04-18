#!/usr/bin/env bash
# Run PD exit-site ResNet sweeps; logs to experiments/*.log and appends results.tsv
set -uo pipefail
cd "$(dirname "$0")/.."
COMMIT=$(git rev-parse --short HEAD)
RESULTS=results.tsv
if [[ ! -f "$RESULTS" ]]; then
  printf 'commit\tmc_acc\tbin_acc\tmemory_gb\tstatus\tdescription\n' > "$RESULTS"
fi

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
  mkdir -p experiments
  echo "========== $name ==========" | tee "$log"
  if python3 train.py --seed 42 "$@" 2>&1 | tee -a "$log"; then
    append_row keep "$name" "$log"
  else
    append_row crash "$name" "$log"
  fi
  echo "" | tee -a "$log"
}

# Fixed 5-minute training budget; high epoch cap so time budget binds first
COMMON=(--epochs 999 --max-train-seconds 300 --log-every 10)

run_exp "e01_baseline_sgd" "${COMMON[@]}" --arch baseline --optimizer sgd --model-out experiments/model_e01.pt
run_exp "e02_wide_sgd" "${COMMON[@]}" --arch wide --optimizer sgd --model-out experiments/model_e02.pt
run_exp "e03_deep_sgd" "${COMMON[@]}" --arch deep --optimizer sgd --model-out experiments/model_e03.pt
run_exp "e04_lite_sgd" "${COMMON[@]}" --arch lite --optimizer sgd --model-out experiments/model_e04.pt
run_exp "e05_baseline_adamw" "${COMMON[@]}" --arch baseline --optimizer adamw --lr 0.0003 --model-out experiments/model_e05.pt
run_exp "e06_deep_bs8" "${COMMON[@]}" --arch deep --batch-size 8 --lr 0.012 --model-out experiments/model_e06.pt

# Optional refinements on the best backbone: ./experiments/run_followup.sh

echo "Done. Results in $RESULTS"
cat "$RESULTS"
