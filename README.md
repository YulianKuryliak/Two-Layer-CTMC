# Two-Layer CTMC

Simulate stochastic epidemics on networks of connected communities with Python.
Choose **Full Gillespie** on the physical network or **TwoScale**, an approximate
MicroMacro model using aggregate inter-community rates and uniform import placement.
Both support SIR and SIS; set the recovery rate to zero for SI dynamics.

## Install

Python **3.10+** is required. The only runtime dependencies are NumPy and NetworkX.

```bash
git clone --branch main --single-branch --depth 1 https://github.com/YulianKuryliak/Two-Layer-CTMC.git
cd Two-Layer-CTMC
python -m venv .venv
```

Activate with `.\.venv\Scripts\Activate.ps1` in Windows PowerShell or
`source .venv/bin/activate` on Linux/macOS, then install:

```bash
python -m pip install .
```

Select this `.venv` as the Python interpreter in VS Code.
Without activation on Windows, use `.\.venv\Scripts\python.exe -m pip install .`
and `.\.venv\Scripts\python.exe -m two_layer_ctmc ...` directly.

## Run from the terminal

```bash
two-layer-ctmc --method full --model SIR --seed 123 --output results/full.csv
two-layer-ctmc --method twoscale --model SIR --seed 123 --output results/twoscale.csv
```

`python -m two_layer_ctmc` provides the same command. Use `--help` for parameters
such as `--communities`, `--community-size`, `--beta`, `--gamma`, `--horizon`,
`--dt` (Full output interval) and `--tau` (TwoScale synchronization interval).
Defaults create three communities of 20 nodes and simulate to time 50.

CSV columns are `community,time,S,I,R`: susceptible, infected and recovered
counts in each community. Full results use a regular output grid; TwoScale
results use its recorded observation times. Existing files are protected;
use a new output path or explicitly add `--overwrite`.

## Python API

```python
from two_layer_ctmc import build_two_layer_network, simulate_micro, simulate_micromacro

network = build_two_layer_network(
    n_communities=3, community_size=20, inter_links=2,
    macro_graph_type="chain", seed=7,
)
full = simulate_micro(
    network=network, beta=0.2, gamma=0.1, model="SIR",
    T_end=50.0, dt_out=1.0, initial_node=0, seed=123,
)
times, infected, communities, events = simulate_micromacro(
    network=network, beta_micro=0.2, gamma=0.1, model="SIR",
    T_end=50.0, tau_micro=0.5, initial_node=0, seed=123,
)
```

`full["rows"]` contains `(community, time, S, I, R)` tuples. TwoScale returns
observation times, total infected counts, per-community `times/S/I/R` lists,
and event records. Results remain in memory unless you save them.
See [examples/run_comparison.py](examples/run_comparison.py) for a complete example.

Full Gillespie samples the network CTMC. TwoScale introduces rate, placement
and synchronization approximations; matching seeds does not make the trajectories
equivalent. Choose parameters and check accuracy for your network and question.

## Tests and license

Run `python -m unittest discover -s tests` after installation.
Version **1.1**. All rights reserved; see [LICENSE](LICENSE).
