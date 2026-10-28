"""Analyze saved national graph and export static figures; no CTMC runs."""
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from prepare_network import HERE, load_sparse, sha256


def main():
    output = HERE / "figures"
    output.mkdir(exist_ok=True)
    manifest_path = output / "manifest.json"
    fingerprint = sha256(__file__)
    input_hash = sha256(HERE / "prepared/manifest.json")
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["source_sha256"] != fingerprint or old["input_manifest_sha256"] != input_hash:
            raise SystemExit("Changed source/input: preserve this output and select a new figure directory")
        for name, digest in old["artifact_sha256"].items():
            assert sha256(output / name) == digest
        print("Verified national figures; skipped")
        return
    nodes, dep, A, W = load_sparse()
    summary = json.loads((HERE / "prepared/summary.json").read_text())
    F = sparse.load_npz(HERE / "prepared/F_domestic.npz")
    f_off = F.copy()
    f_off.setdiag(0)
    f_off.eliminate_zeros()
    imbalance = float(abs(f_off-f_off.T).sum() / (2*f_off.sum()))
    degree = nodes.degree.to_numpy()
    community_size = dep.nodes.to_numpy()
    tri = sparse.triu(A, k=1).tocoo()
    community = nodes.community.to_numpy()
    cross = community[tri.row] != community[tri.col]
    boundary_strength = np.bincount(np.r_[tri.row[cross], tri.col[cross]],
                                    weights=np.r_[tri.data[cross], tri.data[cross]], minlength=len(nodes))
    ranked_boundary = np.sort(boundary_strength[boundary_strength > 0])[::-1]
    top1 = max(1, int(np.ceil(len(nodes)*.01)))
    metrics = dict(
        directed_offdiagonal_imbalance=imbalance,
        raw_rows_under_200_fraction=summary["quality"]["raw_rows_under_200"] / summary["quality"]["raw_rows"],
        raw_worker_weight_under_200_fraction=summary["quality"]["raw_worker_weight_under_200"] / summary["quality"]["raw_total_estimated_workers"],
        outside_worker_fraction=summary["quality"]["outside_estimated_workers"] / summary["quality"]["raw_total_estimated_workers"],
        degree_min=int(degree.min()), degree_median=float(np.median(degree)), degree_max=int(degree.max()),
        department_size_min=int(community_size.min()), department_size_median=float(np.median(community_size)),
        department_size_max=int(community_size.max()), boundary_municipalities=int((boundary_strength > 0).sum()),
        top_1percent_nodes_share_of_boundary_strength=float(ranked_boundary[:top1].sum()/ranked_boundary.sum()),
        dense_A_GiB=summary["dense_full_adjacency_bytes"]/1024**3,
        sparse_A_MiB=summary["sparse_A_bytes"]/1024**2,
        simulations_run=False,
    )
    (output / "analysis.json").write_text(json.dumps(metrics, indent=2) + "\n")
    # Exact chart data and block choice remain inspectable in CSV.
    selected = np.argsort(W.sum(axis=1))[-15:][::-1]
    pd.DataFrame(W[np.ix_(selected, selected)]*summary["flow_scale"],
                 index=dep.iloc[selected].department, columns=dep.iloc[selected].department).to_csv(output / "selected_department_weights.csv")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(8.2, 7), constrained_layout=True)
    weights = W[np.ix_(selected, selected)]*summary["flow_scale"]
    values = np.ma.masked_where(np.eye(15, dtype=bool), np.log10(1+weights))
    im = ax.imshow(values, cmap="Blues", vmin=0)
    codes = dep.iloc[selected].department.tolist()
    ax.set(xticks=range(15), yticks=range(15), xticklabels=codes, yticklabels=codes,
           title="Зв'язки 15 департаментів із найбільшою зовнішньою вагою",
           xlabel="Код департаменту", ylabel="Код департаменту")
    fig.colorbar(im, ax=ax, label="log₁₀(1 + симетрична сума оцінок працівників)")
    fig.savefig(output / "department_links.pdf")
    fig.savefig(output / "department_links.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), constrained_layout=True)
    positive_degrees = np.sort(degree[degree > 0])
    axes[0].step(positive_degrees, np.arange(1,len(positive_degrees)+1)/len(positive_degrees), where="post", color="#315f8c")
    axes[0].set(xscale="log", ylim=(0,1), xlabel="Кількість сусідніх комун (логарифмічна шкала)",
                ylabel="Частка комун серед тих, що мають ребра")
    axes[0].text(.04,.92,f"{summary['isolates']} ізольованих комун поза CDF",transform=axes[0].transAxes,fontsize=9)
    largest = dep.nlargest(12, "nodes").sort_values("nodes")
    axes[1].barh(largest.department, largest.nodes, color="#315f8c")
    axes[1].set(xlabel="Кількість комун у департаменті", ylabel="Код департаменту")
    for ax in axes:
        ax.spines[["right","top"]].set_visible(False)
        ax.grid(axis="x",alpha=.15)
        ax.set_axisbelow(True)
    fig.savefig(output / "national_structure.pdf")
    fig.savefig(output / "national_structure.png", dpi=160)
    plt.close(fig)
    # A stable table fragment supplies the document with empirical top links.
    top = pd.read_csv(HERE / "prepared/top_links.csv", dtype={"code_u":str,"code_v":str}).head(8)
    def tex(value):
        return str(value).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")
    table = [r"\begin{tabular}{llr}", r"\toprule", r"Комуна 1 & Комуна 2 & $F_{uv}+F_{vu}$\\", r"\midrule"]
    for r in top.itertuples():
        table.append(f"{tex(r.name_u)} & {tex(r.name_v)} & {r.worker_connectivity:,.1f}".replace(",", r"\,") + r"\\")
    table += [r"\bottomrule", r"\end{tabular}"]
    (output / "top_links.tex").write_text("\n".join(table)+"\n")
    manifest_path.write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), source_sha256=fingerprint,
        input_manifest_sha256=input_hash, matplotlib=matplotlib.__version__, simulations_run=False,
        artifact_sha256={p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()}), indent=2)+"\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
