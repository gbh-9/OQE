#!/usr/bin/env sh
set -eu
python scripts/run_oqe_v6_event_loss.py
python scripts/run_oqe_v12_followup.py
