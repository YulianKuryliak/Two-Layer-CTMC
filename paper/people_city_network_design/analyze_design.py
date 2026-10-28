"""Deterministic input/partition audit. Does not generate people or run a CTMC."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd
from scipy import sparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREPARED = HERE.parent / "national_network_preparation/prepared"
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_composition(path):
    """Read this official workbook without an optional spreadsheet dependency."""
    with zipfile.ZipFile(path) as outer:
        name, = [n for n in outer.namelist() if n.endswith(".xlsx")]
        with zipfile.ZipFile(io.BytesIO(outer.read(name))) as z:
            strings = ["".join(item.itertext()) for item in
                       ET.fromstring(z.read("xl/sharedStrings.xml"))]
            book = ET.fromstring(z.read("xl/workbook.xml"))
            sheet, = [s for s in book.find("s:sheets", NS)
                      if s.attrib["name"] == "Composition_communale"]
            rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
            target, = [r.attrib["Target"] for r in rels if r.attrib["Id"] == rid]
            member = target.lstrip("/") if target.startswith("/") else "xl/" + target
            rows = []
            header = None
            for row in ET.fromstring(z.read(member)).findall("s:sheetData/s:row", NS):
                cells = {}
                for cell in row:
                    col = "".join(c for c in cell.attrib["r"] if c.isalpha())
                    value = cell.find("s:v", NS)
                    value = "" if value is None else value.text
                    if cell.attrib.get("t") == "s" and value:
                        value = strings[int(value)]
                    cells[col] = value
                if "CODGEO" in cells.values():
                    header = cells
                elif header is not None:
                    record = {key: cells.get(col, "") for col, key in header.items()}
                    if len(record.get("CODGEO", "")) == 5:
                        rows.append(record)
    frame = pd.DataFrame(rows)
    if frame.CODGEO.duplicated().any():
        raise ValueError("Duplicate commune membership")
    return frame


def aggregate_flows(flow, labels):
    codes, names = pd.factorize(labels, sort=True)
    membership = sparse.csr_matrix((np.ones(len(codes)), (np.arange(len(codes)), codes)),
                                   shape=(len(codes), len(names)))
    phi = (membership.T @ flow @ membership).tocsr()
    np.testing.assert_allclose(phi.sum(), flow.sum(), rtol=1e-12)
    return phi, codes, names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "analysis/v2")
    args = parser.parse_args()
    inputs = [HERE / "sources.json", HERE / "raw/population_2022_geography2025.zip",
              HERE / "raw/urban_units_2025.zip", PREPARED / "nodes.csv",
              PREPARED / "F_domestic.npz"]
    fingerprints = {str(p.relative_to(ROOT)): digest(p) for p in inputs + [Path(__file__)]}
    if args.output.exists():
        manifest = json.loads((args.output / "manifest.json").read_text())
        if manifest["fingerprints"] != fingerprints:
            raise SystemExit("Changed input/code; use a NEW --output directory")
        for name, sha in manifest["artifacts"].items():
            if digest(args.output / name) != sha:
                raise SystemExit("Artifact hash mismatch: " + name)
        print("Verified immutable audit; skipped existing output")
        return
    started = datetime.now(timezone.utc).isoformat()
    source_manifest = json.loads((HERE / "sources.json").read_text())
    for item in source_manifest["downloaded"]:
        if digest(HERE / item["path"]) != item["sha256"]:
            raise ValueError("Downloaded source hash mismatch")
    nodes = pd.read_csv(PREPARED / "nodes.csv", dtype={"commune_code": str,
                        "department": str, "region": str})
    if nodes.commune_code.duplicated().any():
        raise ValueError("Duplicate baseline commune")
    ages = ["P22_POP0014", "P22_POP1529", "P22_POP3044", "P22_POP4559",
            "P22_POP6074", "P22_POP7589", "P22_POP90P"]
    with zipfile.ZipFile(inputs[1]) as z:
        with z.open("base-cc-evol-struct-pop-2022.CSV") as handle:
            population = pd.read_csv(handle, sep=";", usecols=["CODGEO", "P22_POP"] + ages,
                                     dtype={"CODGEO": str})
    composition = read_composition(inputs[2])
    joined = nodes.merge(population, left_on="commune_code", right_on="CODGEO",
                         how="left", validate="one_to_one").drop(columns="CODGEO")
    joined = joined.merge(composition[["CODGEO", "UU2020", "LIBUU2020", "TYPE_COMMUNE_UU"]],
                          left_on="commune_code", right_on="CODGEO", how="left",
                          validate="one_to_one").drop(columns="CODGEO")
    if joined[["UU2020", "TYPE_COMMUNE_UU"]].isna().any().any():
        raise ValueError("Missing geographic join")
    if set(nodes.commune_code) - set(population.CODGEO):
        raise ValueError("Missing population row; distinguish from present but NA values")
    joined["population_available"] = joined[["P22_POP"] + ages].notna().all(axis=1)
    missing = joined.loc[~joined.population_available, ["commune_code", "name"]]
    # Preserve missing source values as NA. Never interpret them as empty communes.
    if set(missing.commune_code) != {"15031", "15035", "15047", "15171"}:
        raise ValueError("Unexpected missing population fields; investigate before proceeding")
    np.testing.assert_array_equal(joined.node_id, np.arange(len(nodes)))
    numeric = joined.loc[joined.population_available, ["P22_POP"] + ages].to_numpy()
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise ValueError("Invalid population")
    # Official age counts are rounded separately, so retain and report discrepancies.
    joined["age_margin_difference"] = joined[ages].sum(axis=1, min_count=len(ages)) - joined.P22_POP
    joined["workers_minus_population15plus"] = (
        joined.estimated_resident_workers_in_source - (joined.P22_POP - joined.P22_POP0014))
    rural = joined.TYPE_COMMUNE_UU == "Hors unité urbaine"
    joined["partition_urban_rural_department"] = np.where(
        rural, "R:" + joined.department, "U:" + joined.UU2020)
    joined["partition_urban_rural_commune"] = np.where(
        rural, "R:" + joined.commune_code, "U:" + joined.UU2020)
    flow = sparse.load_npz(PREPARED / "F_domestic.npz").tocsr()
    if flow.shape != (len(nodes), len(nodes)) or not np.isfinite(flow.data).all() or (flow.data < 0).any():
        raise ValueError("Invalid directed commuting matrix")
    specs = [("commune", joined.commune_code),
             ("urban_rural_commune", joined.partition_urban_rural_commune),
             ("urban_rural_department", joined.partition_urban_rural_department)]
    stats = []
    groups = []
    for name, labels in specs:
        phi, codes, names = aggregate_flows(flow, labels)
        total_population = np.bincount(codes, weights=joined.P22_POP.fillna(0))
        missing_members = np.bincount(codes, weights=~joined.population_available)
        stats.append(dict(partition=name, groups=len(names),
            same_group_worker_fraction=float(phi.diagonal().sum() / phi.sum()),
            largest_available_population_sum=float(total_population.max()),
            groups_with_missing_population=int((missing_members > 0).sum()),
            dense_macro_lower_bound_gib=5 * 8 * len(names)**2 / 2**30))
        groups.append(pd.DataFrame(dict(partition=name, group=names,
            available_population_sum=total_population, communes=np.bincount(codes),
            missing_population_members=missing_members.astype(int))))
    # A possible regional first case; do not redistribute the excluded flows.
    mask = joined.region.eq("53").to_numpy()
    within = float(flow[mask][:, mask].sum())
    outbound = float(flow[mask].sum()) - within
    inbound = float(flow[:, mask].sum()) - within
    external_destination = float(joined.loc[mask, "estimated_resident_workers_in_source"].sum()
                                 - flow[mask].sum())
    total_population = float(joined.P22_POP.sum())
    summary = dict(kind="deterministic_data_and_partition_audit", simulations_run=False,
        synthetic_people_generated=0, seeds=None, commune_count=len(nodes),
        population_source_rows=len(population), urban_composition_rows=len(composition),
        available_population_sum=float(total_population),
        missing_population_communes=missing.to_dict(orient="records"),
        age_margin_max_abs=float(joined.age_margin_difference.abs().max()),
        age_margin_nonzero_communes=int(joined.age_margin_difference.dropna().abs().gt(1e-8).sum()),
        workers_exceed_population15plus_communes=int(joined.workers_minus_population15plus.gt(1e-8).sum()),
        max_worker_excess=float(joined.workers_minus_population15plus.max()),
        rural_communes=int(rural.sum()), rural_population=float(joined.loc[rural, "P22_POP"].sum()),
        urban_units=int(joined.loc[~rural, "UU2020"].nunique()),
        domestic_workers=float(flow.sum()), partitions=stats,
        illustrative_csr_degree20_gib=(total_population * 20 * 12 + 8 * (total_population + 1))/2**30,
        csr_note="Hypothetical average degree20 on available population sum, int32 neighbor+float64 weight+int64 offsets; one adjacency only. Four commune rows have NA population values in the downloaded source; no imputation was performed; totals are sums over available values.",
        bretagne=dict(communes=int(mask.sum()), population=float(joined.loc[mask, "P22_POP"].sum()),
            workers_within_region=within, workers_to_other_domestic_regions=outbound,
            workers_from_other_domestic_regions=inbound,
            resident_workers_to_outside_domestic_scope=external_destination),
        notes=["F/Phi count estimated workers, never contact weights W.",
               "R:department is an auxiliary rural group, not a city.",
               "No outside-resident origins in this source scope.",
               "Five dense arrays give a conservative lower bound, not peak RAM or timing."])
    args.output.mkdir(parents=True, exist_ok=False)
    joined.to_csv(args.output / "commune_inputs.csv", index=False)
    pd.concat(groups, ignore_index=True).to_csv(args.output / "partition_groups.csv", index=False)
    pd.DataFrame(stats).to_csv(args.output / "partition_summary.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    labels = {"commune": "Комуни", "urban_rural_commune": "Агломерації + сільські комуни",
              "urban_rural_department": "Агломерації + сільські групи департаментів"}
    table = [r"\begin{tabularx}{\linewidth}{Xrrr}\toprule",
             r"Поділ & $M$ & Внутрішні поїздки, \% & 5 масивів, GiB\\\midrule"]
    for row in stats:
        table.append(f"{labels[row['partition']]} & {row['groups']} & "
                     f"{100*row['same_group_worker_fraction']:.2f} & "
                     f"{row['dense_macro_lower_bound_gib']:.2f}" + r"\\")
    table.append(r"\bottomrule\end{tabularx}")
    (args.output / "partition_table.tex").write_text("\n".join(table) + "\n")
    artifacts = {p.name: digest(p) for p in sorted(args.output.iterdir())}
    manifest = dict(start_utc=started, end_utc=datetime.now(timezone.utc).isoformat(),
        fingerprints=fingerprints, artifacts=artifacts, simulations_run=False,
        numpy=np.__version__, pandas=pd.__version__, status="complete")
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
