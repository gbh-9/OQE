#!/usr/bin/env sh
set -eu
python scripts/benchmark_oqe_v6_suffix.py --mode microbench
python scripts/benchmark_oqe_v6_suffix.py --mode full
