#!/bin/bash
# Overnight master runner for Two-Layer-CTMC pending experiments.
# Runs stages SEQUENTIALLY (each stage internally parallel via --workers) to
# avoid CPU oversubscription. Each stage is --resume'd, so partial progress
# from earlier interrupted runs is preserved and continued, never lost.
# A stage failure does NOT stop the run -- next stages still execute.
#
# Usage:
#   nohup bash extrascripts/runners/run_overnight_experiments.sh > /tmp/overnight_master.log 2>&1 &
#   disown
# or simply run it in an attached terminal that stays open overnight.

set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python

STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="output/logs/overnight_${STAMP}"
mkdir -p "$LOGDIR"
MASTER_LOG="$LOGDIR/master.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"; }

trap 'log "!!! INTERRUPTED (signal) !!!"' INT TERM

run_stage() {
    local name="$1"; shift
    local stage_log="$LOGDIR/${name}.log"
    log "START ${name} :: $*"
    if "$@" > "$stage_log" 2>&1; then
        log "DONE  ${name} (ok)"
    else
        log "DONE  ${name} (FAILED rc=$? -- see $stage_log)"
    fi
}

log "=== Overnight run started ==="
log "Logs directory: $LOGDIR"

# 1. Finish CE-06 regime map full grid (was at 25/36 cells)
run_stage ce06_full \
    "$PY" extrascripts/runners/run_ce06_regime_map.py --grid full --workers 8 --resume

# 2. CE-10 correction ladder: finish pilot (was 4/6), then run full grid
run_stage ce10_pilot \
    "$PY" extrascripts/runners/run_ce10_correction_ladder.py --grid pilot --workers 4 --resume
run_stage ce10_full \
    "$PY" extrascripts/runners/run_ce10_correction_ladder.py --grid full --workers 8 --resume

# 3. ER-01 multi-topology smoke (validates correction across topologies,
#    key requirement: must not overfit to a single chain-of-cliques topology)
run_stage er01_smoke \
    "$PY" extrascripts/runners/run_er01_multitopo_smoke.py --grid smoke --workers 2 --resume

# 4. CE-07 calibration floor (structural vs calibration error separation)
run_stage ce07_smoke \
    "$PY" extrascripts/runners/run_ce07_calibration_floor.py --grid smoke --workers 4 --resume
run_stage ce07_pilot \
    "$PY" extrascripts/runners/run_ce07_calibration_floor.py --grid pilot --workers 4 --resume
run_stage ce07_full \
    "$PY" extrascripts/runners/run_ce07_calibration_floor.py --grid full --workers 8 --resume

# 5. CE-08 negative control (clique-bridge-leaf sanity check)
run_stage ce08_smoke \
    "$PY" extrascripts/runners/run_ce08_negative_control.py --grid smoke --workers 2 --resume
run_stage ce08_full \
    "$PY" extrascripts/runners/run_ce08_negative_control.py --grid full --workers 8 --resume

# 6. CE-09 boundary/gateway isolation
run_stage ce09_smoke \
    "$PY" extrascripts/runners/run_ce09_boundary_isolation.py --grid smoke --workers 4 --resume
run_stage ce09_pilot \
    "$PY" extrascripts/runners/run_ce09_boundary_isolation.py --grid pilot --workers 4 --resume
run_stage ce09_full \
    "$PY" extrascripts/runners/run_ce09_boundary_isolation.py --grid full --workers 8 --resume

log "=== Overnight run finished (all stages attempted) ==="
