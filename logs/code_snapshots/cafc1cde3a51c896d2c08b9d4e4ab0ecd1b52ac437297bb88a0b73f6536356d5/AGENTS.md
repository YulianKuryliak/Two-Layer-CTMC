## Automatic Planning And Subagent Routing

The main Codex session is the planner/orchestrator. Subagents are bounded workers, not the main decision-maker.

Codex should automatically choose the cheapest adequate workflow for each task.

### Trivial tasks

Examples:
- answer a narrow question;
- fix one typo;
- inspect one known file;
- make a tiny local edit.

Workflow:
- Do not use subagents.
- Do not use heavy planning.
- Work directly and briefly.

### Simple single-file tasks

Examples:
- edit one paragraph;
- adjust one caption;
- check one LaTeX label;
- make one localized code change.

Workflow:
- Do not use subagents by default.
- Produce a short 2-4 step written plan only if the edit may affect scientific meaning, correctness, reproducibility, or build behavior.
- Edit only the target file.
- Run only targeted verification if needed.

### Medium tasks

Examples:
- locate where a concept appears in the paper;
- compare a paper section with existing figures;
- inspect saved benchmark outputs;
- check LaTeX references across several files.

Workflow:
- Produce a short written plan before editing.
- Use at most one cheap read-only subagent if it saves context or time.
- Prefer:
  - paper_searcher for repository/text search;
  - data_checker for saved benchmark outputs;
  - latex_checker for labels, references, figure paths, and bibliography keys.
- Do not use write-capable agents unless the needed edit is clear.

### Large paper/research tasks

Examples:
- rewrite or reorganize a section;
- review approximation-error logic;
- evaluate benchmark validity;
- connect results, figures, and claims;
- prepare a major Results/Discussion change.

Workflow:
- Start with a written plan before editing.
- Use cheap read-only subagents first when subtasks are bounded and independent.
- Use scientific_reviewer only for scientific judgment, benchmark design, mathematical interpretation, or final prose quality.
- Summarize subagent findings before editing.
- Ask for confirmation before large edits.
- Use paper_writer only for approved localized prose edits.

### Experiment, plotting, or code execution tasks

Examples:
- run scripts;
- regenerate plots;
- inspect simulation outputs;
- check reproducibility.

Workflow:
- First check whether saved data already exists.
- Prefer data_checker before experiment_runner.
- Prefer plot_maker only when figures must be regenerated from saved data.
- Do not rerun simulations unless explicitly approved by the user.
- Run small pilots, smoke tests, and short simulation batches directly when their cost is modest.
- For large or long simulation suites, prefer preparing a reproducible runner for the user instead of occupying the main Codex session with execution and waiting.
- A delegated runner must specify scenarios, parameters, deterministic seeds, output directories, full-trajectory policy, and completion metadata. It should support resume/skip-existing behavior when practical.
- Give the user one clear launch command and state the expected outputs. Do not start the large run unless the user explicitly asks Codex to run it.
- Before handing a large run to the user, preserve enough context in the runner, manifest, plan, or project state file to continue without repeating the scientific analysis.
- When the user reports that runs are complete, inspect the saved outputs, validate completeness against the manifest, and continue the existing analysis and paper task from that point.
- Do not delegate a run merely to avoid a few inexpensive commands; use this workflow only when runtime, scenario count, or monitoring overhead makes delegation worthwhile.

### True plan mode

- True Codex CLI `/plan` mode is controlled by the user.
- Do not assume true `/plan` mode can be automatically toggled.
- If true `/plan` mode is already enabled, follow it.
- If true `/plan` mode is not enabled, emulate planning behavior with a written plan only for medium, large, experiment, plotting, or code-execution tasks.
- Do not force heavy planning for trivial or simple tasks.

### Economy rules

- Do not launch the full subagent workflow by default.
- Do not spawn multiple subagents unless their tasks are clearly independent.
- Do not use subagents for trivial or simple single-file tasks.
- For medium tasks, use at most one cheap read-only subagent when useful.
- For large tasks, use multiple read-only subagents only when the subtasks are independent and the benefit justifies the overhead.
- Prefer cheap read-only agents before stronger or write-capable agents.
- Use stronger agents only when scientific correctness or paper quality requires it.
- Use write-capable agents only after the plan is clear.
- Do not allow multiple write-capable agents to edit the same file in parallel.
- If unsure whether subagents are worth it, do not use them.
- Do not sacrifice correctness, reproducibility, or paper quality just to save usage.

### Model routing

- Use `gpt-5.6-luna` for cheap bounded search, data inspection, LaTeX checks, and mechanical edits.
- Use `gpt-5.6-luna` with low reasoning for bounded execution and efficient monitoring of already specified commands; run trivial commands directly without a subagent.
- Monitoring agents must only report process status, failures, and output completeness. They must not reinterpret scientific results or alter experiment parameters.
- If a cheaper configured model than `gpt-5.6-luna` is available and is adequate for pure status polling or file-existence checks, prefer it; otherwise use `gpt-5.6-luna` low.
- Use `gpt-5.6-terra` for nontrivial code review, experiment design or failure diagnosis, and plot generation.
- Use `gpt-5.6-sol` only for scientific judgment, mathematical interpretation, benchmark design, and approved paper-level writing.
- Keep reasoning effort low for mechanical/read-only work and medium for scientific or implementation work unless a task clearly requires otherwise.
- If a configured model is unavailable, use the closest available model with the same cost/quality role and report the substitution.

### Runtime fallback

- To use project custom agents from `.codex/agents/*.toml`, start Codex with custom-agent support enabled when this CLI requires it, for example with `--enable multi_agent_v2`.
- The `codex-ctmc` shell shortcut launches this project through `start-codex-paper.sh` with `danger-full-access`, `on-request`, and `multi_agent_v2` enabled.
- The equivalent direct command is `codex -C /mnt/c/Users/UAKuryliYu/MyFolder/Personal/Projects/Two-Layer-CTMC -s danger-full-access -a on-request --enable multi_agent_v2`.
- If configured custom agent names such as `paper_searcher` or `data_checker` are not exposed by the current Codex runtime, use the built-in `explorer` role for read-only bounded checks and the built-in `worker` role only for explicitly delegated write tasks.
- When using built-in roles as a fallback, pass the intended model and reasoning level explicitly when possible, and include the same scope limits that the matching custom agent would have used.
- Treat fallback agents as subject to the same economy, write-access, parallelism, and no-simulation rules.

## Parallel Process Policy

- Codex and delegated subagents may run parallel local processes when the tasks are independent and parallelism saves meaningful time.
- Prefer parallel reads and inspections, such as `rg`, `rg --files`, `sed`, `ls`, `git status`, lightweight CSV inspection, and LaTeX path checks.
- Keep parallelism bounded. Do not exceed the configured subagent/thread limit when one exists; otherwise use a small number of concurrent tasks.
- Do not run multiple write-capable processes against the same file or output directory in parallel.
- Do not run simulations, expensive benchmarks, or plot regeneration in parallel unless explicitly approved by the user.
- For long-running local processes, track what was started, wait for required processes before final reporting, and report failures or incomplete runs.
- Subagents may run local commands only within their delegated scope and must not launch nested subagents.
- Prefer saved data and existing outputs over launching new compute jobs.

## Paper Focus

- The current paper should focus on approximation error, not runtime speedup.
- Runtime optimization can be mentioned as future work, but new claims should primarily evaluate when the two-scale approximation is accurate or biased relative to the full-network Gillespie CTMC.
- Important error sources include rate abstraction error, import placement error, and synchronization/numerical error.
- Important comparisons include Full Gillespie, basic two-scale MicroMacro, bridge-aware MicroMacro, oracle-rate variants, and oracle-placement variants.
- Completion probability, censoring, arrival-time distributions, survival/CDF curves, and cumulative arrival errors must be handled carefully.
- Do not compare results across different horizons or censoring rules as if they were the same experiment.

## Write Access Rules

- Read-only agents should never edit files.
- Write-capable agents may edit only files explicitly delegated by the parent Codex session.
- Large paper edits require a written plan first.
- Large simulation runs require explicit approval.
- Plot regeneration should use saved data when possible.
- After write operations, report changed files and suggested verification.
- Before final reporting, inspect git diff when available.

## Mandatory Code And Experiment History

- Current user preference for the corrected paper rerun: Codex prepares and
  analyzes scripts; the user launches them. Do not start simulation runners,
  diagnostic experiments or smoke simulations without a new explicit request.

- Log every task that changes code in `history.md` and append a source snapshot
  record to `logs/code_changes.jsonl` using
  `.venv/bin/python extrascripts/analysis/log_code_change.py --message "..." --verification "..." <changed files>`.
- Include every changed source file, runner and test. Record scientific behavior,
  verification and limitations; never silently fold unrelated pre-existing edits
  into the claimed change. Do not rewrite earlier records.
- Log every new simulation/experiment invocation, including diagnostics and
  smoke runs, in the separate `logs/experiments.jsonl`. Record start, parameters,
  seeds, output, code fingerprint, end/status, failures and resume/skip decisions.
- `extrascripts/runners/rerun_smpt_paper.py` writes experiment records automatically
  to the central log and its output folder. Other runners must explicitly record
  equivalent lifecycle metadata before execution.
- Keep old results immutable. A changed code fingerprint requires a new output
  directory; it must not silently resume a previous scientific experiment.
