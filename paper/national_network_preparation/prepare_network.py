"""Build the national INSEE municipality network. Deterministic; no CTMC runs.

Canonical representation: sparse observed directed F, symmetric proxy A,
department partition and dense department W (not municipality-sized dense W).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent
FLOW = "NBFLUX_C22_ACTOCC15P"
RAW = ["mobility_2022.zip", "v_commune_2025.csv", "v_departement_2025.csv"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def construct_layers(rows, cols, values, owner, n_communities, flow_scale):
    """Sum legitimate post-crosswalk duplicates; self-flows stay descriptive.

    A is a symmetric contact proxy, not a person-contact measurement. W derives
    from A, never directly from directed F, and its diagonal is always zero.
    """
    if not np.isfinite(flow_scale) or flow_scale <= 0:
        raise ValueError("flow_scale must be finite and positive")
    values = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("flows must be finite and positive")
    owner = np.asarray(owner, dtype=np.int64)
    if len(owner) == 0 or owner.min() < 0 or owner.max() >= n_communities:
        raise ValueError("Invalid community partition")
    F = sparse.coo_matrix((values, (rows, cols)), shape=(len(owner), len(owner))).tocsr()
    F.sum_duplicates()
    A = ((F + F.T) / flow_scale).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    coo = F.tocoo()
    descriptive = np.zeros((n_communities, n_communities), dtype=float)
    np.add.at(descriptive, (owner[coo.row], owner[coo.col]), coo.data)
    W = (descriptive + descriptive.T) / flow_scale
    np.fill_diagonal(W, 0)
    return F, A, W, descriptive


def load_sparse(folder=HERE / "prepared"):
    folder = Path(folder)
    nodes = pd.read_csv(folder / "nodes.csv", dtype={"commune_code": str, "department": str, "region": str})
    departments = pd.read_csv(folder / "departments.csv", dtype={"department": str, "region": str})
    return (nodes, departments, sparse.load_npz(folder / "A_proxy.npz"), np.load(folder / "W.npy"))


def load_networkx(folder=HERE / "prepared"):
    """Explicit optional memory expansion for the existing undirected APIs.

    Local graphs are read-only views of one full graph, with integer node IDs.
    No epidemic engine, spectral eigensolver, event log or trajectory is created.
    """
    import networkx as nx
    nodes, departments, A, W = load_sparse(folder)
    full = nx.from_scipy_sparse_array(A, create_using=nx.Graph, edge_attribute="weight")
    nx.set_node_attributes(full, dict(zip(nodes.node_id, nodes.commune_code)), "commune_code")
    groups = [nodes.loc[nodes.community == i, "node_id"].astype(int).tolist() for i in range(len(departments))]
    local = [full.subgraph(group) for group in groups]
    full.graph["community_sizes"] = [len(group) for group in groups]
    full.graph["alphas"] = [1.0] * len(groups)
    full.graph["network_semantics"] = "municipality nodes; symmetric commuting-based spatial diffusion proxy"
    return local, full, W, groups


def prepare(raw, output, flow_scale):
    cog = pd.read_csv(raw / "v_commune_2025.csv", dtype=str).fillna("")
    dep = pd.read_csv(raw / "v_departement_2025.csv", dtype=str).fillna("")
    communes = cog.loc[(cog.TYPECOM == "COM") & (cog.DEP != "976")].copy()
    if communes.COM.duplicated().any():
        raise ValueError("Duplicate commune codes in COG")
    dep = dep.loc[dep.DEP.isin(communes.DEP)].sort_values("DEP").reset_index(drop=True)
    dep_index = {code: i for i, code in enumerate(dep.DEP)}
    communes = communes.sort_values(["DEP", "COM"]).reset_index(drop=True)
    node_index = {code: i for i, code in enumerate(communes.COM)}
    crosswalk = {row.COM: row.COM for row in cog.loc[cog.TYPECOM == "COM"].itertuples()}
    for row in cog.loc[cog.TYPECOM == "ARM"].itertuples():
        if row.COMPARENT not in crosswalk:
            raise ValueError(f"ARM parent missing: {row.COM}")
        crosswalk[row.COM] = row.COMPARENT
    with zipfile.ZipFile(raw / "mobility_2022.zip") as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError("Expected exactly one mobility CSV")
        with archive.open(members[0]) as handle:
            flows = pd.read_csv(handle, sep=";", dtype={"CODGEO": str, "DCLT": str})
    if flows[["CODGEO", "DCLT", FLOW]].isna().any().any():
        raise ValueError("Missing required flow field")
    if flows.duplicated(["CODGEO", "DCLT"]).any():
        raise ValueError("Duplicate original OD key; inspect before aggregating")
    if not np.isfinite(flows[FLOW]).all() or (flows[FLOW] <= 0).any():
        raise ValueError("Invalid raw flows")
    origin = flows.CODGEO.map(crosswalk)
    destination = flows.DCLT.map(crosswalk)
    if origin.isna().any() or not origin.isin(node_index).all():
        raise ValueError("Unexpected source geography: needs an explicit policy")
    keep = destination.isin(node_index)
    outside = flows.loc[~keep].copy()
    outside["reason"] = np.where(destination[~keep].isna(), "not_in_COG_commune_ARM_dictionary", "outside_scope_Mayotte")
    outside_stats = outside.groupby("reason")[FLOW].agg(["size", "sum"]).reset_index()
    all_origin_totals = flows.groupby(origin)[FLOW].sum()
    resident_workers = communes.COM.map(all_origin_totals).fillna(0).to_numpy()
    rows = origin[keep].map(node_index).to_numpy(dtype=np.int32)
    cols = destination[keep].map(node_index).to_numpy(dtype=np.int32)
    values = flows.loc[keep, FLOW].to_numpy(dtype=float)
    owner = communes.DEP.map(dep_index).to_numpy(dtype=np.int32)
    F, A, W, descriptive = construct_layers(rows, cols, values, owner, len(dep), flow_scale)
    output.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(output / "F_domestic.npz", F)
    sparse.save_npz(output / "A_proxy.npz", A)
    np.save(output / "W.npy", W)
    np.save(output / "department_directed_flows_descriptive.npy", descriptive)
    pd.DataFrame(W, index=dep.DEP, columns=dep.DEP).to_csv(output / "W.csv", index_label="department")
    pd.DataFrame(descriptive, index=dep.DEP, columns=dep.DEP).to_csv(output / "department_directed_flows_descriptive.csv", index_label="department")
    outside.to_csv(output / "outside_scope_flows.csv.gz", index=False)
    coo = sparse.triu(A, k=1).tocoo()
    inside = owner[coo.row] == owner[coo.col]
    boundary = np.zeros(len(A.indptr)-1, dtype=bool)
    boundary[coo.row[~inside]] = True
    boundary[coo.col[~inside]] = True
    strengths = np.asarray(A.sum(axis=1)).ravel()
    nodes = pd.DataFrame(dict(node_id=np.arange(len(communes)), commune_code=communes.COM,
                              name=communes.LIBELLE, department=communes.DEP, region=communes.REG,
                              community=owner, estimated_resident_workers_in_source=resident_workers,
                              same_commune_workers=F.diagonal(), boundary=boundary.astype(int),
                              degree=np.diff(A.indptr), strength=strengths))
    nodes.to_csv(output / "nodes.csv", index=False)
    component_count, component_labels = connected_components(A, directed=False)
    comp_sizes = np.bincount(component_labels)
    comp_table = pd.DataFrame(dict(component=np.arange(component_count), nodes=comp_sizes))
    comp_table.to_csv(output / "components.csv", index=False)
    community_rows = []
    for i, record in enumerate(dep.itertuples()):
        ids = np.flatnonzero(owner == i)
        local = A[ids][:, ids]
        local_components, local_labels = connected_components(local, directed=False)
        internal_weight = float(local.sum()/2)
        external_weight = float(W[i].sum())
        community_rows.append(dict(community=i, department=record.DEP, name=record.LIBELLE, region=record.REG,
                                    nodes=len(ids), internal_edges=local.nnz//2, components=local_components,
                                    largest_component=int(np.bincount(local_labels).max()),
                                    local_isolates=int((np.diff(local.indptr) == 0).sum()),
                                    full_isolates=int((nodes.loc[ids, "degree"] == 0).sum()),
                                    boundary_nodes=int(boundary[ids].sum()), internal_weight=internal_weight,
                                    external_weight=external_weight,
                                    internal_strength_fraction=(2*internal_weight/(2*internal_weight+external_weight)
                                                                if 2*internal_weight+external_weight else None)))
    pd.DataFrame(community_rows).to_csv(output / "departments.csv", index=False)
    # Preserve the crosswalk independently of all node renumbering.
    pd.DataFrame([dict(raw_code=k, commune_code=v, in_scope=int(v in node_index))
                  for k, v in sorted(crosswalk.items())]).to_csv(output / "geography_crosswalk.csv", index=False)
    edge_count = len(coo.data)
    total_inter_weight = float(coo.data[~inside].sum())
    domestic_self = float(F.diagonal().sum())
    domestic_offdiag = float(F.sum()-domestic_self)
    assert np.allclose(W, W.T) and np.all(np.diag(W) == 0)
    assert np.isclose(W.sum()/2, total_inter_weight)
    assert np.isclose(coo.data.sum()*flow_scale, domestic_offdiag)
    assert np.isclose(float(F.sum())+float(outside[FLOW].sum()), float(flows[FLOW].sum()))
    assert (A-A.T).nnz == 0 and np.all(A.diagonal() == 0)
    assert np.isclose(sum(r["internal_weight"] for r in community_rows) + W.sum()/2, A.sum()/2)
    top = np.argsort(coo.data)[-15:][::-1]
    pd.DataFrame([dict(u=int(coo.row[k]), v=int(coo.col[k]),
                            code_u=nodes.iloc[coo.row[k]].commune_code,
                            name_u=nodes.iloc[coo.row[k]]["name"],
                            code_v=nodes.iloc[coo.col[k]].commune_code,
                            name_v=nodes.iloc[coo.col[k]]["name"],
                            worker_connectivity=float(coo.data[k]*flow_scale)) for k in top]).to_csv(output / "top_links.csv", index=False)
    quality = dict(raw_rows=len(flows), raw_total_estimated_workers=float(flows[FLOW].sum()),
                    duplicate_original_OD_keys=0, missing_flow_fields=0, nonpositive_flows=0,
                    original_origins=int(flows.CODGEO.nunique()), original_destinations=int(flows.DCLT.nunique()),
                    domestic_rows=int(keep.sum()), outside_rows=int((~keep).sum()),
                    outside_estimated_workers=float(outside[FLOW].sum()),
                    outside_categories=outside_stats.to_dict("records"),
                    rows_using_ARM=int(((flows.CODGEO != origin) | (flows.DCLT != destination)) .where(keep, False).sum()),
                    post_crosswalk_domestic_OD_pairs=F.nnz, domestic_same_commune_workers=domestic_self,
                    domestic_distinct_commune_workers=domestic_offdiag,
                    raw_rows_under_200=int((flows[FLOW] < 200).sum()),
                    raw_worker_weight_under_200=float(flows.loc[flows[FLOW] < 200, FLOW].sum()),
                    COG_scope_nodes_without_origin_rows=int((resident_workers == 0).sum()))
    summary = dict(nodes=len(nodes), undirected_edges=edge_count, communities=len(dep),
                    geography="France metropolitan and four overseas departments, excluding Mayotte, COG 2025; ARM collapsed to parent COM",
                    model="Municipality-level SI diffusion on symmetric commuting-based proxy; not a person-level epidemic model",
                    flow_scale=flow_scale, weight_definition="A_uv=(F_uv+F_vu)/flow_scale for u!=v; W from interdepartment sums of A; diag(W)=0",
                    intra_edges=int(inside.sum()), inter_edges=int((~inside).sum()),
                    within_edge_weight_fraction=float(coo.data[inside].sum()/coo.data.sum()),
                    components=component_count, largest_component=int(comp_sizes.max()),
                    isolates=int((np.diff(A.indptr)==0).sum()),
                    departments_with_disconnected_local_graph=sum(r["components"] > 1 for r in community_rows),
                    nonempty_macro_pairs=int(np.count_nonzero(np.triu(W, 1))),
                    max_department_nodes=max(r["nodes"] for r in community_rows),
                    dense_full_adjacency_bytes=len(nodes)**2*8,
                    sparse_A_bytes=A.data.nbytes+A.indices.nbytes+A.indptr.nbytes,
                    dense_macro_W_bytes=W.nbytes, conservation_checks="PASS", simulations_run=False,
                    quality=quality)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "prepared")
    parser.add_argument("--flow-scale", type=float, default=1000.0)
    args = parser.parse_args()
    output = args.output.resolve()
    inputs = {name: sha256(HERE / "raw" / name) for name in RAW}
    fingerprint = sha256(__file__)
    provenance = dict(source_sha256=fingerprint, input_sha256=inputs, flow_scale=args.flow_scale)
    log_path = HERE / "preparation_runs.jsonl"
    def record(event, **kwargs):
        with log_path.open("a") as handle:
            handle.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), event=event,
                                        kind="deterministic_network_preparation", output=str(output),
                                        seeds=None, simulations_run=False, **provenance, **kwargs)) + "\n")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if any(old[k] != v for k, v in provenance.items()):
            raise SystemExit("Changed inputs/code/config: select a NEW --output directory")
        for name, digest in old["artifact_sha256"].items():
            if sha256(output / name) != digest:
                raise SystemExit(f"Artifact changed: {name}")
        record("skip_existing", reason="matching fingerprints; all artifact hashes verified")
        print("Verified existing national network; skipped without rewriting")
        return
    if output.exists() and any(output.iterdir()):
        raise SystemExit("Output must be empty or have a completed matching manifest")
    record("start")
    started = datetime.now(timezone.utc).isoformat()
    try:
        result = prepare(HERE / "raw", output, args.flow_scale)
        manifest_path.write_text(json.dumps(dict(**provenance, start_utc=started,
            end_utc=datetime.now(timezone.utc).isoformat(),
            numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__,
            source_manifest_sha256=sha256(HERE / "sources.json"), seeds=None, simulations_run=False,
            artifact_sha256={p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()}), indent=2) + "\n")
    except Exception as exc:
        record("failed", error=repr(exc))
        raise
    record("complete", nodes=result["nodes"], edges=result["undirected_edges"])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
