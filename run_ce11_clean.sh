#!/bin/bash
set -e
cd /mnt/c/Users/UAKuryliYu/MyFolder/Personal/Projects/Two-Layer-CTMC
echo "Running CE-11 analysis..."
.venv/bin/python extrascripts/analysis/analyze_ce11_horizon.py
echo "Done."
