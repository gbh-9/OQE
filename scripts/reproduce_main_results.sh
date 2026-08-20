#!/usr/bin/env sh
set -eu
python scripts/prepare_datasets.py
python scripts/run_oqe_v6_main.py --dataset all --arm all --include-non-nested --batch-size 128
python scripts/summarize_oqe_v6.py
