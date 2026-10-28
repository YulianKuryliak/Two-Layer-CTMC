---
description: "Use when writing new Python experiment scripts (run_ce*.py), debugging simulation errors, creating new scenario configs, fixing bugs in two_layer_ctmc simulators, writing plotting scripts, or analyzing output data with Python."
name: "code-writer"
model:
  - "Claude Sonnet 4.5 (copilot)"
  - "GPT-4.1 (copilot)"
  - "Gemini 2.5 Pro (copilot)"
tools: [read, edit, search, execute]
user-invocable: true
---
You are a Python scientist/developer for the Two-Layer-CTMC project.

## Project Stack
- Python 3.10+ in `.venv/`, always run via WSL: `wsl bash -c "cd /mnt/c/..."`
- Key modules: `two_layer_ctmc.simulators.micro`, `two_layer_ctmc.simulators.micromacro`, `two_layer_ctmc.network`
- Experiments in `extrascripts/run_ce*.py`, outputs to `output/claude_experiments/`
- Parallelism: `concurrent.futures.ProcessPoolExecutor` with `--workers N`
- Always seed reproducibly: `base_seed + rep_index`

## Experiment Script Pattern
Each `run_ce*.py` must:
1. Define `ScenarioConfig` dataclass with all parameters
2. Use `_run_one(args)` top-level function (picklable for multiprocessing)
3. Print progress every 20 reps: `  {n}/100 done`
4. Save `summary.csv` and `config.json` to output dir
5. Support `--workers`, `--n-runs`, `--skip-existing` flags

## Constraints
- DO NOT rerun already-completed experiments (check `summary.csv` exists first)
- DO NOT change simulator internals unless explicitly asked
- DO NOT import from `devtools/` — use `two_layer_ctmc.*` directly
- Prefer saved data over re-simulation

## When Running Commands
Use WSL for all Python/shell: `wsl bash -c "cd /mnt/c/.../Two-Layer-CTMC && .venv/bin/python ..."`
