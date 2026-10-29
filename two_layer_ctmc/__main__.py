"""Command-line simulation with CSV output."""
import argparse
import csv
import math
from pathlib import Path
from typing import Optional, Sequence

from . import build_two_layer_network, simulate_micro, simulate_micromacro


def _positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and greater than zero")
    return number


def _nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate SIR/SIS on connected communities and save CSV counts."
    )
    parser.add_argument("--method", choices=("full", "twoscale"), default="full",
                        help="full Gillespie or approximate TwoScale (default: full)")
    parser.add_argument("--model", choices=("SIR", "SIS"), default="SIR")
    parser.add_argument("--communities", type=_positive_int, default=3)
    parser.add_argument("--community-size", type=_positive_int, default=20)
    parser.add_argument("--inter-links", type=_nonnegative_int, default=2)
    parser.add_argument("--macro-topology", choices=("complete", "chain"), default="chain")
    parser.add_argument("--micro-topology", choices=("complete", "chain"), default="complete")
    parser.add_argument("--beta", type=_nonnegative_float, default=0.2,
                        help="infection rate (default: 0.2)")
    parser.add_argument("--gamma", type=_nonnegative_float, default=0.1,
                        help="recovery rate; zero gives SI (default: 0.1)")
    parser.add_argument("--horizon", type=_positive_float, default=50.0)
    parser.add_argument("--dt", type=_positive_float, default=1.0,
                        help="Full output sampling interval (default: 1)")
    parser.add_argument("--tau", type=_positive_float, default=0.5,
                        help="TwoScale synchronization interval (default: 0.5)")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--initial-node", type=_nonnegative_int, default=0,
                        help="initially infected node in community 0 (default: 0)")
    parser.add_argument("--output", type=Path, default=Path("simulation.csv"))
    parser.add_argument("--overwrite", action="store_true",
                        help="allow replacing an existing output file")
    return parser


def _rows(method: str, result):
    if method == "full":
        yield from result["rows"]
    else:
        _, _, logs, _ = result
        for community, data in logs.items():
            for time, susceptible, infected, recovered in zip(
                data["times"], data["S"], data["I"], data["R"]
            ):
                yield community, time, susceptible, infected, recovered


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output}; choose a new path or use --overwrite")
    try:
        network = build_two_layer_network(
            n_communities=args.communities,
            community_size=args.community_size,
            inter_links=args.inter_links,
            seed=args.seed,
            macro_graph_type=args.macro_topology,
            micro_graph_type=args.micro_topology,
        )
        common = dict(network=network, gamma=args.gamma, model=args.model,
                      T_end=args.horizon, initial_node=args.initial_node, seed=args.seed)
        if args.method == "full":
            result = simulate_micro(beta=args.beta, dt_out=args.dt, **common)
        else:
            result = simulate_micromacro(beta_micro=args.beta, tau_micro=args.tau, **common)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w" if args.overwrite else "x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("community", "time", "S", "I", "R"))
            writer.writerows(_rows(args.method, result))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Saved {args.method} {args.model} counts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
