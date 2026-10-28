"""Prepare SocioPatterns contacts for SMPT; never run epidemic simulations.

Run from the repository root:
  .venv/bin/python paper/network_preparation/prepare_network.py
The loader returns micro_graphs, full_graph, W, comm_nodes. Population defaults
to observed contact endpoints; metadata-only persons are an explicit alternative.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import sys

import networkx as nx
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAGE = "https://sociopatterns.org/datasets/high-school-contact-and-friendship-networks/"
URLS = {
    "HighSchool2013_proximity_net.csv.gz": "https://sociopatterns.org/assets/data/HighSchool2013_proximity_net.csv.gz",
    "HighSchool2013_metadata.txt": "https://sociopatterns.org/assets/data/HighSchool2013_metadata.txt",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_metadata(path):
    metadata = {}
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"Metadata line {line_no}: expected three fields")
        node = int(fields[0])
        if node in metadata:
            raise ValueError(f"Duplicate metadata ID: {node}")
        metadata[node] = fields[1]
    return metadata


def aggregate_contacts(path, metadata):
    """One canonical unordered pair per 20-second bin; duplicates are errors."""
    counts, bins, day_bins = Counter(), set(), defaultdict(set)
    rows, seen, active = 0, set(), set()
    contact_labels, disagreements = {}, Counter()
    for line_no, line in enumerate(gzip.open(path, "rt"), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Contact line {line_no}: expected five fields")
        t, u, v = map(int, fields[:3])
        if u == v:
            raise ValueError(f"Self-contact at line {line_no}")
        if u not in metadata or v not in metadata:
            raise ValueError(f"Contact endpoint missing metadata at line {line_no}")
        for node, label in zip((u, v), fields[3:]):
            if node in contact_labels and contact_labels[node] != label:
                raise ValueError(f"Time-varying class for ID {node}; explicit policy required")
            contact_labels[node] = label
            if metadata[node] != label:
                disagreements[node, metadata[node], label] += 1
        u, v = sorted((u, v))
        key = (t, u, v)
        if key in seen:
            raise ValueError(f"Duplicate unordered contact-bin: {key}")
        seen.add(key)
        counts[u, v] += 1
        active.update((u, v))
        bins.add(t)
        day_bins[datetime.fromtimestamp(t, timezone.utc).date().isoformat()].add(t)
        rows += 1
    if not rows:
        raise ValueError("Empty contacts")
    if len({t % 20 for t in bins}) != 1:
        raise ValueError("Contact timestamps do not share a 20-second grid")
    tmin, tmax = min(bins), max(bins)
    # Source intervals are [t-20, t]. Include both endpoint bins exactly once.
    duration = tmax - tmin + 20
    quality = dict(
        rows=rows, unique_contact_bins=rows, duplicate_contact_bins=0,
        metadata_nodes=len(metadata), observed_nodes=len(active),
        metadata_only_ids=sorted(set(metadata) - active),
        missing_metadata=0, class_mismatch_endpoints=sum(disagreements.values()), self_contacts=0,
        class_disagreements=[dict(node=u, metadata_class=m, contact_class=c, endpoint_rows=n)
                             for (u, m, c), n in sorted(disagreements.items())],
        class_policy="Separate metadata table governs primary partition; contact-label alternative exported explicitly.",
        timestamp_min=tmin, timestamp_max=tmax,
        observation_start_utc=datetime.fromtimestamp(tmin - 20, timezone.utc).isoformat(),
        observation_end_utc=datetime.fromtimestamp(tmax, timezone.utc).isoformat(),
        observation_seconds=duration, active_bins=len(bins),
        total_pair_contact_seconds=20 * rows,
        bins_per_day={day: len(values) for day, values in sorted(day_bins.items())},
        note="A contact-bin row is not an episode; consecutive rows can belong to one episode. Empty bins are not distinguished from missing observation.",
    )
    return counts, active, quality, contact_labels


def split_graph(graph, labels):
    """Preserve graph topology; each undirected external weight enters W twice."""
    classes = sorted({labels[u] for u in graph})
    comm_nodes = [sorted(u for u in graph if labels[u] == label) for label in classes]
    owner = {u: i for i, nodes in enumerate(comm_nodes) for u in nodes}
    W = np.zeros((len(classes), len(classes)))
    boundary = [set() for _ in classes]
    for u, v, data in graph.edges(data=True):
        w = float(data["weight"])
        if not np.isfinite(w) or w <= 0:
            raise ValueError("All retained weights must be positive and finite")
        i, j = owner[u], owner[v]
        if i != j:
            W[i, j] += w
            W[j, i] += w
            boundary[i].add(u)
            boundary[j].add(v)
    micro = [graph.subgraph(nodes).copy() for nodes in comm_nodes]
    graph.graph["community_sizes"] = [len(nodes) for nodes in comm_nodes]
    # Baseline loader: alpha/correction modes must remain off. No new correction
    # is inferred from empirical topology and no bridge identities enter W.
    graph.graph["alphas"] = [1.0] * len(classes)
    graph.graph["boundary_counts"] = [len(nodes) for nodes in boundary]
    return micro, graph, W, comm_nodes


def load_prepared(folder=HERE / "prepared", *, population="observed", weight_mode="duration", class_source="metadata"):
    """Load simulator inputs, without constructing or running an epidemic model.

    duration: weight=d_uv/T_obs; binary: weight=1 for every observed pair.
    metadata: retain the two metadata-only persons as isolates (sensitivity).
    """
    if population not in {"observed", "metadata"}:
        raise ValueError("population must be observed or metadata")
    if weight_mode not in {"duration", "binary"}:
        raise ValueError("weight_mode must be duration or binary")
    if class_source not in {"metadata", "contacts"}:
        raise ValueError("class_source must be metadata or contacts")
    folder = Path(folder)
    graph, labels = nx.Graph(), {}
    with (folder / "nodes.csv").open() as handle:
        for row in csv.DictReader(handle):
            u = int(row["node"])
            if population == "metadata" or int(row["observed"]):
                labels[u] = (row["contact_class"] or row["class"]) if class_source == "contacts" else row["class"]
                graph.add_node(u, school_class=labels[u])
    with (folder / "edges.csv").open() as handle:
        for row in csv.DictReader(handle):
            graph.add_edge(int(row["u"]), int(row["v"]),
                           weight=float(row["weight"]) if weight_mode == "duration" else 1.0)
    return split_graph(graph, labels)


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def graph_descriptors(micro, full, W, comm_nodes):
    classes = [full.nodes[nodes[0]]["school_class"] for nodes in comm_nodes]
    owner = {u: i for i, nodes in enumerate(comm_nodes) for u in nodes}
    interfaces, communities = [], []
    for i, local in enumerate(micro):
        adjacency = nx.to_numpy_array(local, nodelist=comm_nodes[i], weight="weight")
        strength = adjacency.sum(axis=1)
        inv = np.zeros_like(strength)
        inv[strength > 0] = 1 / np.sqrt(strength[strength > 0])
        lap = np.diag((strength > 0).astype(float)) - inv[:, None] * adjacency * inv[None, :]
        mu2 = max(0.0, float(np.linalg.eigvalsh(lap)[1])) if len(local) > 1 else None
        boundary = {u for u in local if any(owner[v] != i for v in full[u])}
        internal_weight = local.size(weight="weight")
        external_weight = float(W[i].sum())
        communities.append(dict(
            community=i, school_class=classes[i], nodes=len(local), edges=local.number_of_edges(),
            components=nx.number_connected_components(local), density=nx.density(local),
            aspl=nx.average_shortest_path_length(local) if nx.is_connected(local) else None,
            mu2_weighted=mu2, boundary_nodes=len(boundary), boundary_fraction=len(boundary) / len(local),
            internal_weight=internal_weight, external_weight=external_weight,
            internal_strength_fraction=2 * internal_weight / (2 * internal_weight + external_weight),
        ))
        for j in range(i + 1, len(classes)):
            links = [(u, v, d["weight"]) for u, v, d in full.edges(data=True)
                     if {owner[u], owner[v]} == {i, j}]
            if not links:
                continue
            a, b = Counter(), Counter()
            for u, v, w in links:
                if owner[u] == j:
                    u, v = v, u
                a[u] += w
                b[v] += w
            total = float(W[i, j])
            # Exact L1 distance between normalized empirical cross-weights and
            # the uniform surrogate, including all absent cross-pairs as zeros.
            pairs = len(comm_nodes[i]) * len(comm_nodes[j])
            tv = 0.5 * (sum(abs(w / total - 1 / pairs) for _, _, w in links)
                        + (pairs - len(links)) / pairs)
            interfaces.append(dict(
                i=i, j=j, class_i=classes[i], class_j=classes[j], edges=len(links), W=total,
                boundary_i=len(a), boundary_j=len(b),
                boundary_fraction_i=len(a) / len(comm_nodes[i]),
                boundary_fraction_j=len(b) / len(comm_nodes[j]),
                effective_endpoints_i=total**2 / sum(w*w for w in a.values()),
                effective_endpoints_j=total**2 / sum(w*w for w in b.values()),
                total_variation_from_uniform=tv,
            ))
    return communities, interfaces


def prepare(raw, output):
    metadata = read_metadata(raw / "HighSchool2013_metadata.txt")
    counts, active, quality, contact_labels = aggregate_contacts(raw / "HighSchool2013_proximity_net.csv.gz", metadata)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "nodes.csv", ["node", "class", "contact_class", "observed"],
              [dict(node=u, **{"class": metadata[u]}, contact_class=contact_labels.get(u, ""),
                    observed=int(u in active)) for u in sorted(metadata)])
    write_csv(output / "edges.csv", ["u", "v", "contact_bins", "duration_seconds", "weight"],
              [dict(u=u, v=v, contact_bins=c, duration_seconds=20*c,
                    weight=20*c / quality["observation_seconds"]) for (u, v), c in sorted(counts.items())])
    micro, full, W, comm_nodes = load_prepared(output)
    communities, interfaces = graph_descriptors(micro, full, W, comm_nodes)
    classes = [row["school_class"] for row in communities]
    write_csv(output / "communities.csv", list(communities[0]), communities)
    write_csv(output / "interfaces.csv", list(interfaces[0]), interfaces)
    write_csv(output / "W.csv", ["class"] + classes,
              [dict(zip(["class"] + classes, [label] + list(W[i]))) for i, label in enumerate(classes)])
    binary = load_prepared(output, weight_mode="binary")[2]
    write_csv(output / "W_binary.csv", ["class"] + classes,
              [dict(zip(["class"] + classes, [label] + list(binary[i]))) for i, label in enumerate(classes)])
    alt_micro, alt_full, alt_W, alt_nodes = load_prepared(output, class_source="contacts")
    alt_comms, alt_interfaces = graph_descriptors(alt_micro, alt_full, alt_W, alt_nodes)
    write_csv(output / "communities_contact_labels.csv", list(alt_comms[0]), alt_comms)
    write_csv(output / "W_contact_labels.csv", ["class"] + classes,
              [dict(zip(["class"] + classes, [label] + list(alt_W[i]))) for i, label in enumerate(classes)])
    (output / "community_nodes.json").write_text(json.dumps(dict(zip(classes, comm_nodes)), indent=2) + "\n")
    # GraphML cannot serialize list-valued graph metadata; metadata is restored
    # by load_prepared. This copy is only a portable topology/weight export.
    exported = full.copy()
    exported.graph.clear()
    nx.write_graphml(exported, output / "full_graph.graphml")
    for i, local in enumerate(micro):
        nx.write_graphml(local, output / f"local_{i}.graphml")
    cross = sum(d["weight"] for u, v, d in full.edges(data=True) if metadata[u] != metadata[v])
    inside = sum(g.size(weight="weight") for g in micro)
    assert np.allclose(W, W.T) and np.all(np.diag(W) == 0)
    assert np.isclose(W.sum() / 2, cross)
    assert np.isclose(inside + cross, full.size(weight="weight"))
    assert sum(counts.values()) == quality["rows"]
    assert np.isclose(full.size(weight="weight") * quality["observation_seconds"], 20*quality["rows"])
    summary = dict(
        quality=quality, population_policy="Observed endpoints (327); all 329 metadata IDs retained in nodes.csv for an explicit sensitivity alternative.",
        weighting="duration_seconds / observation_seconds; calendar time including nights and gaps; beta in inverse seconds of active contact.",
        nodes=len(full), edges=full.number_of_edges(), communities=len(micro),
        components=nx.number_connected_components(full),
        intra_edges=sum(g.number_of_edges() for g in micro), inter_edges=sum(r["edges"] for r in interfaces),
        intra_weight=inside, inter_weight=cross, within_weight_fraction=inside/(inside+cross),
        nonempty_interfaces=len(interfaces), possible_interfaces=len(micro)*(len(micro)-1)//2,
        contact_label_alternative=dict(
            intra_edges=sum(g.number_of_edges() for g in alt_micro),
            inter_edges=sum(r["edges"] for r in alt_interfaces),
            inter_weight=float(alt_W.sum()/2),
            W_frobenius_relative_difference=float(np.linalg.norm(alt_W-W)/np.linalg.norm(W)),
            W_max_absolute_difference=float(np.max(np.abs(alt_W-W))),
        ),
        conservation_checks="PASS", simulations_run=False,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "prepared")
    args = parser.parse_args()
    raw = HERE / "raw"
    inputs = {name: sha256(raw / name) for name in URLS}
    fingerprint = sha256(__file__)
    output = args.output.resolve()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["source_sha256"] != fingerprint or old["input_sha256"] != inputs:
            raise SystemExit("Changed code/input: use --output with a NEW directory; old results are immutable.")
        for name, digest in old["artifact_sha256"].items():
            if sha256(output / name) != digest:
                raise SystemExit(f"Saved artifact hash mismatch: {name}")
        print("Verified existing preparation; skipped without rewriting.")
        return
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Output must be empty or have a matching completed manifest")
    started = datetime.now(timezone.utc).isoformat()
    log = HERE / "preparation_runs.jsonl"
    def record(event, **extra):
        with log.open("a") as handle:
            handle.write(json.dumps(dict(event=event, kind="deterministic_data_preparation", utc=datetime.now(timezone.utc).isoformat(),
                                         source_sha256=fingerprint, input_sha256=inputs, output=str(output),
                                         seeds=None, simulations_run=False, **extra)) + "\n")
    record("start")
    try:
        summary = prepare(raw, output)
        manifest = dict(start_utc=started, end_utc=datetime.now(timezone.utc).isoformat(), source_sha256=fingerprint,
                        input_sha256=inputs, source_urls=URLS, landing_page=PAGE,
                        python=sys.version, platform=platform.platform(), networkx=nx.__version__, numpy=np.__version__,
                        seeds=None, simulations_run=False,
                        artifact_sha256={p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()})
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    except Exception as exc:
        record("failed", error=repr(exc))
        raise
    record("complete", nodes=summary["nodes"], edges=summary["edges"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
