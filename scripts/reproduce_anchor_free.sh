#!/usr/bin/env sh
set -eu
python scripts/run_oqe_anchor_models.py --batch-size 128
python scripts/run_oqe_v18_zero_anchor.py --phase all --batch-size 128
python scripts/run_oqe_anchor_erasure_corrected.py --batch-size 128
