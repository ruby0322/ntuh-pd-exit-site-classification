#!/usr/bin/env bash
# Focused sweep around e02_wide_sgd (bin_acc ~0.896, mc_acc ~0.568): lr / wd / momentum / Nesterov.
set -uo pipefail
cd "$(dirname "$0")/.."
COMMIT=$(git rev-parse --short HEAD)
RESULTS=results.tsv
mkdir -p experiments

BEST_BIN=0.896186
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

run_exp "e11_wide_lr012" "${COMMON[@]}" --lr 0.012 --model-out experiments/model_e11.pt
run_exp "e12_wide_lr008" "${COMMON[@]}" --lr 0.008 --model-out experiments/model_e12.pt
run_exp "e13_wide_wd5e5" "${COMMON[@]}" --weight-decay 0.00005 --model-out experiments/model_e13.pt
run_exp "e14_wide_wd2e4" "${COMMON[@]}" --weight-decay 0.0002 --model-out experiments/model_e14.pt
run_exp "e15_wide_mom095" "${COMMON[@]}" --momentum 0.95 --model-out experiments/model_e15.pt
run_exp "e16_wide_nesterov" "${COMMON[@]}" --nesterov --model-out experiments/model_e16.pt

echo "Done. Tail of $RESULTS:"
tail -n12 "$RESULTS"
