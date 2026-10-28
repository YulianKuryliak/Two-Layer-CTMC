#!/usr/bin/env bash
# Sequential, resume-safe launcher for the four previously missing experiments.
# Each simulation runner parallelizes internally; stages remain sequential to
# avoid CPU oversubscription and preserve deterministic output directories.

set -uo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="output/logs/missing_experiments_${STAMP}"
mkdir -p "$LOGDIR"
MASTER_LOG="$LOGDIR/master.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER_LOG"; }

run_stage() {
    local name="$1"
    shift
    log "START ${name} :: $*"
    if "$@" >"$LOGDIR/${name}.log" 2>&1; then
        log "DONE  ${name} (ok)"
    else
        local rc=$?
        log "DONE  ${name} (FAILED rc=${rc}; see $LOGDIR/${name}.log)"
    fi
}

log "=== Missing-experiment queue started ==="
log "Logs: $LOGDIR"

# Analysis-only; uses saved CE-01 and CE-06 data.
run_stage ce12_power "$PY" extrascripts/runners/run_ce12_power_calibration.py \
    --bootstrap-reps 500 --resume \
    --output-dir output/claude_experiments/ce12_power_calibration

# Reuses archived Full/baseline arms and simulates only two new compact arms.
run_stage oracle_bridge "$PY" extrascripts/runners/run_oracle_bridge_ceiling.py \
    --grid full --workers 8 --resume \
    --output-dir output/missing_experiments/oracle_bridge_ceiling

# Reuses archived Full/event-driven arms and runs production tau refinement.
run_stage synchronization "$PY" extrascripts/runners/run_synchronization_audit.py \
    --grid full --workers 8 --resume \
    --output-dir output/missing_experiments/synchronization_audit

# New paired near/random/far network experiment; MicroMacro is reused across modes.
run_stage gateway_distance "$PY" extrascripts/runners/run_gateway_distance_sweep.py \
    --grid full --workers 8 \
    --output-dir output/missing_experiments/gateway_distance/full

log "=== Missing-experiment queue finished ==="
