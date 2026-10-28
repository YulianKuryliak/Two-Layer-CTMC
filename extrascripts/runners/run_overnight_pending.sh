#!/bin/bash
# Overnight runner for the REMAINING pending experiments (2026-08-04).
# Stages run SEQUENTIALLY (each internally parallel via --workers) to avoid
# CPU oversubscription on the 16-core machine. Ordered cheap+important first,
# most-expensive last. Every stage is --resume'd, so partial progress from an
# interrupted run is continued, never re-simulated. A stage failure does NOT
# stop the queue.
#
# Launch (survives terminal/VS Code closure):
#   wsl bash -c "cd <repo> && tmux new-session -d -s overnight 'bash extrascripts/runners/run_overnight_pending.sh'"
# Check:  wsl tmux capture-pane -t overnight -p | tail -30
#         output/logs/overnight_<STAMP>/master.log

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

log "=== Overnight PENDING run started ==="
log "Logs directory: $LOGDIR"

# 1. CE-09 boundary/gateway isolation (explains why C2 breaks on clique_leaves).
run_stage ce09_smoke "$PY" extrascripts/runners/run_ce09_boundary_isolation.py --grid smoke --workers 6  --resume
run_stage ce09_pilot "$PY" extrascripts/runners/run_ce09_boundary_isolation.py --grid pilot --workers 12 --resume
run_stage ce09_full  "$PY" extrascripts/runners/run_ce09_boundary_isolation.py --grid full  --workers 12 --resume

# 2. CE-08 negative control (cheap; falsifiability / defensibility).
run_stage ce08_full  "$PY" extrascripts/runners/run_ce08_negative_control.py --grid full --workers 12 --resume

# 3. ER-01 multi-topology PILOT (CIs across topologies; resolves the C2/C12
#    clique_leaves regression -- THE key open question).
run_stage er01_pilot "$PY" extrascripts/runners/run_er01_multitopo_smoke.py --grid pilot --workers 12 --resume

# 4. CE-07 calibration floor FULL (most expensive; last, resumable so partial
#    progress persists to a future night).
run_stage ce07_full  "$PY" extrascripts/runners/run_ce07_calibration_floor.py --grid full --workers 12 --resume

log "=== Overnight PENDING run finished (all stages attempted) ==="
