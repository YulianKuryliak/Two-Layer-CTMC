"""Plot saved prepared topology and evaluate one fixed-state rate witness.

No trajectories, epidemic experiments, random seeds or simulation calls.
"""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from prepare_network import HERE, load_prepared, sha256


def main():
    output = HERE / "figures"
    output.mkdir(exist_ok=True)
    manifest_path = output / "manifest.json"
    fingerprint = sha256(__file__)
    prepared_hash = sha256(HERE / "prepared/manifest.json")
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["source_sha256"] != fingerprint or old["prepared_manifest_sha256"] != prepared_hash:
            raise SystemExit("Changed source/input: preserve this folder and choose a new output in the script")
        for name, digest in old["artifact_sha256"].items():
            assert sha256(output / name) == digest
        print("Verified existing static figures; skipped.")
        return
    with (HERE / "prepared/communities.csv").open() as handle:
        communities = list(csv.DictReader(handle))
    with (HERE / "prepared/interfaces.csv").open() as handle:
        interfaces = list(csv.DictReader(handle))
    local, full, W, nodes = load_prepared()
    labels = [row["school_class"] for row in communities]
    # A witness is a deterministic, valid initial state: one infected source u,
    # every other node susceptible. Compare exact and aggregate i->j hazards.
    candidates = []
    for i, group in enumerate(nodes):
        for j, target in enumerate(nodes):
            if i == j or W[i, j] == 0:
                continue
            target_set = set(target)
            for u in group:
                if not target_set.intersection(full[u]):
                    candidates.append((float(W[i, j]/len(group)), i, j, u))
    rate, i, j, u = max(candidates)
    witness = dict(infected_node=u, source_class=labels[i], target_class=labels[j],
                   source_size=len(nodes[i]), target_size=len(nodes[j]), W_ij=float(W[i,j]),
                   exact_hazard_divided_by_beta=0.0, macro_hazard_divided_by_beta=rate,
                   state="Only this source node infected; all remaining nodes susceptible.",
                   meaning="One fixed-state counterexample, selected to maximize aggregate hazard among zero-exact-hazard source/target choices. Not a frequency or estimated arrival-time bias.")
    metrics = dict(
        interface_TV_min=min(float(r["total_variation_from_uniform"]) for r in interfaces),
        interface_TV_max=max(float(r["total_variation_from_uniform"]) for r in interfaces),
        interface_TV_median=float(np.median([float(r["total_variation_from_uniform"]) for r in interfaces])),
        witness=witness, simulations_run=False,
    )
    (output / "analysis.json").write_text(json.dumps(metrics, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8, 6.8), constrained_layout=True)
    # Display total observed contact minutes, keeping W dimensionless in CSV.
    duration = json.loads((HERE / "prepared/summary.json").read_text())["quality"]["observation_seconds"]
    matrix = W * duration / 60
    display = np.ma.masked_where(np.eye(len(W), dtype=bool), matrix)
    color = ax.imshow(display, cmap="Blues", vmin=0, vmax=matrix.max())
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
           title="Міжкласні контакти за весь період, хвилини")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(len(W)):
        for j in range(len(W)):
            if i != j:
                ax.text(j, i, f"{matrix[i,j]:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if matrix[i,j] > matrix.max()*0.55 else "#222222")
    fig.colorbar(color, ax=ax, label="Сумарні парні хвилини контакту")
    fig.savefig(output / "macro_contacts.pdf")
    fig.savefig(output / "macro_contacts.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    pos = np.arange(len(labels))
    internal = [100*float(r["internal_strength_fraction"]) for r in communities]
    axes[0].barh(pos, internal, color="#315f8c")
    axes[0].set(yticks=pos, yticklabels=labels, xlim=(0,100), xlabel="Частка внутрішньої сили вузлів, %")
    axes[0].invert_yaxis()
    for y, value in enumerate(internal):
        axes[0].text(value-1, y, f"{value:.1f}", ha="right", va="center", color="white", fontsize=8)
    axes[1].scatter([float(r["edges"]) for r in interfaces],
                    [float(r["total_variation_from_uniform"]) for r in interfaces],
                    color="#315f8c", s=28, alpha=.8)
    axes[1].set(xlabel="Кількість реальних ребер інтерфейсу", ylabel="TV до рівномірних ваг", ylim=(0,1.03))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
    fig.savefig(output / "structure_checks.pdf")
    fig.savefig(output / "structure_checks.png", dpi=180)
    plt.close(fig)
    manifest_path.write_text(json.dumps(dict(
        utc=datetime.now(timezone.utc).isoformat(), source_sha256=fingerprint,
        prepared_manifest_sha256=prepared_hash, matplotlib=matplotlib.__version__,
        kind="deterministic_saved_data_analysis", seeds=None, simulations_run=False,
        artifact_sha256={p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
    ), indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
