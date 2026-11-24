#!/usr/bin/env bash
# Re-runs the OULAD analyses needed for the fourth review round. Usage (from the code folder): bash run_oulad_review4.sh [NJ]
set -euo pipefail
cd "$(dirname "$0")/src"
NJ="${1:-8}"
export OULAD_DATA_DIR="${OULAD_DATA_DIR:-$(cd .. && pwd)/data/oulad}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# 1. main held-out run: primary-seed audit (B=20000), per-seed audits (B=2000), fixed-capacity audit, group tables
python oulad_audit.py --mode heldout --cutoffs 25,50 --out ../results/oulad_heldout --B 20000 --seed_B 2000 --seeds 5 --n_jobs "$NJ"
# 2. same-day random-split CV (threshold tuned on a calibration split); gender x IMD cells now keep unknown IMD separate
python oulad_audit.py --mode cv --cv_thr 1 --cutoffs 25,50,75 --repeats 5 --out ../results/oulad_cv_sameday --cv_targets gender --n_jobs "$NJ" --B 5000
# 3. held-out without score-based features (submission information retained)
python oulad_audit.py --mode heldout --cutoffs 25,50 --out ../results/oulad_heldout_noscores --no_scores 1 --B 5000 --seed_B 0 --seeds 3 --n_jobs "$NJ"
# 4. held-out restricted to students with a recorded IMD band
python oulad_audit.py --mode heldout --cutoffs 25,50 --out ../results/oulad_heldout_knownimd --known_imd 1 --B 5000 --seed_B 0 --seeds 3 --n_jobs "$NJ"
echo "ALL DONE"
