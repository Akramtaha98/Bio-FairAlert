#!/usr/bin/env bash
# One-command runner for the OULAD audits on the H200 (single GPU 0). Usage: bash run_oulad_h200.sh [all|heldout|cv|regen]
set -euo pipefail
STAGE="${1:-all}"
cd "$(dirname "$0")/src"
export OULAD_DATA_DIR="${OULAD_DATA_DIR:-$(cd .. && pwd)/data/oulad}"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # 1 thread per worker; parallelism is across workers
NJ="${NJ:-$(nproc)}"
for f in studentInfo studentRegistration studentAssessment assessments courses; do
  [ -f "$OULAD_DATA_DIR/$f.csv" ] || { echo "MISSING $OULAD_DATA_DIR/$f.csv  (set OULAD_DATA_DIR=...)"; exit 1; }
done
DEV=cpu
python - <<'P' && DEV=cuda || true
import sys
try:
    import torch; sys.exit(0 if torch.cuda.is_available() else 1)
except Exception: sys.exit(1)
P
echo "workers=$NJ  bootstrap device=$DEV  data=$OULAD_DATA_DIR"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
mkdir -p ../logs
if [[ "$STAGE" == all || "$STAGE" == heldout ]]; then
  python oulad_audit.py --mode heldout --cutoffs 25,50 --out ../results/oulad_heldout --B 20000 --seeds 5 \
     --n_jobs "$NJ" --device "$DEV" 2>&1 | tee ../logs/oulad_heldout.log
fi
if [[ "$STAGE" == all || "$STAGE" == cv ]]; then
  python oulad_audit.py --mode cv --cutoffs 75 --repeats 5 --out ../results/oulad_cv_gender --B 20000 \
     --cv_targets gender --n_jobs "$NJ" --device "$DEV" 2>&1 | tee ../logs/oulad_cv_gender.log
fi
if [[ "$STAGE" == all || "$STAGE" == regen ]]; then
  # regenerate the committed OULAD analyses into results_regen/ (independent scripts run in parallel), then diff
  export BIOFAIRALERT_OUT_DIR="$(cd .. && pwd)/results_regen" BIOFAIRALERT_MEDIA_DIR="$(cd .. && pwd)/results_regen/figures"
  mkdir -p "$BIOFAIRALERT_OUT_DIR"
  cp ../results/mi_stacking_partial.json ../results/mooc_mi_stacking_partial.json "$BIOFAIRALERT_OUT_DIR"/ 2>/dev/null || true
  for s in repeated_intersectional repeated_ablation fairness_panel_repeated explanation_stability explain_oulad bootstrap_fairness; do
    ( python "$s.py" > "../logs/regen_$s.log" 2>&1 && echo "regen $s OK" || echo "regen $s FAILED (see logs/regen_$s.log)" ) &
  done
  wait
  python compare_regen.py ../results ../results_regen | tee ../logs/regen_compare.txt
fi
echo "ALL DONE"
