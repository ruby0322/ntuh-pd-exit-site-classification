#!/usr/bin/env bash
# Micro LR bracket around e11 (lr=0.012); best bin 0.923729 mc 0.567797
set -uo pipefail
cd "$(dirname "$0")/.."
COMMIT=$(git rev-parse --short HEAD)
RESULTS=results.tsv
mkdir -p experiments

BEST_BIN=0.923729
BEST_MC=0.567797

append_row() {
  local desc=$1 log=$2
  local mc bin mem_mb mem_gb status
  mc=$(grep '^mc_acc:' "$log" | tail -1 | awk '{print $2}')
  bin=$(grep '^bin_acc:' "$log" | tail -1 | awk '{print $2}')
  mem_mb=$(grep '^peak_vram_mb:' "$log" | tail -1 | awk '{print $2}')
  mem_gb=$(python3 -c "print(round(float('${mem_mb:-0}')/1024, 1))")
  status=$(python3 -c "b=float('${bin:-0}'); m=float('${mc:-0}'); bb=float('$BEST_BIN'); bm=float('$BEST_MC'); \
 print('keep' if (b > bb + 1e-9) or (abs(b-bb) < 1e-6 and m > bm + 1e-9) else 'discard')")
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$COMMIT" "${mc:-0}" "${bin:-0}" "$mem_gb" "$status" "$desc" >> "$RESULTS"
  if [[ "$status" == "keep" ]]; then
    BEST_BIN="$bin"
    BEST_MC="$mc"
  fi
}

run_exp() {
  local name=$1
  shift
  local log="experiments/${name}.log"
  echo "========== $name ==========" | tee "$log"
  if python3 train.py --seed 42 "$@" 2>&1 | tee -a "$log"; then
    append_row "$name" "$log"
  else
    printf '%s\t0\t0\t0.0\tcrash\t%s\n' "$COMMIT" "$name" >> "$RESULTS"
  fi
  echo "" | tee -a "$log"
}

COMMON=(--epochs 999 --max-train-seconds 300 --log-every 10 --arch wide --optimizer sgd)

run_exp "e21_wide_lr0118" "${COMMON[@]}" --lr 0.0118 --model-out experiments/model_e21.pt
run_exp "e22_wide_lr0122" "${COMMON[@]}" --lr 0.0122 --model-out experiments/model_e22.pt

echo "Done. Tail of $RESULTS:"
tail -n6 "$RESULTS"
