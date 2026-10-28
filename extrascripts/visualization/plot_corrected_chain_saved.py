#!/usr/bin/env python3
"""Regenerate the ASPL-chain figures from a completed, saved analysis.

This is an adapter, deliberately kept separate from the publication plotting
scripts.  It redirects their module globals to one analysis directory and
never reads checkpoints or starts a simulation.  The required CSV/NPZ inputs
are made by the corresponding saved-data analysis step.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.util
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
VIS = ROOT / "extrascripts" / "visualization"
EXPECTED_PAIRS = 400
ASPL_VALUES = (1, 2, 3, 4, 5, 6)
K_VALUES = (1, 5)


def load_original(stem: str):
    """Load an unchanged plotting module under its normal import name."""
    path = VIS / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import plotting script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[stem] = module
    spec.loader.exec_module(module)
    return module, path


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved-analysis input: {path}")
    return path


def rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require_complete_layer_metrics(path: Path) -> list[dict[str, str]]:
    result = rows(require(path))
    expected = {(a, k, c) for a in ASPL_VALUES for k in K_VALUES for c in range(1, 10)}
    actual = {(int(r["target_ASPL"]), int(r["k"]), int(r["community"])) for r in result}
    if actual != expected or len(result) != len(expected):
        raise RuntimeError("layer_metrics.csv must contain exactly 108 ASPL × k × community cells")
    censored = [r for r in result if int(r["both_observed"]) != EXPECTED_PAIRS]
    if censored:
        sample = censored[0]
        raise RuntimeError(
            "Refusing to draw ordinary arrival-error figures from censored data: "
            f"ASPL={sample['target_ASPL']}, k={sample['k']}, community={sample['community']} "
            f"has {sample['both_observed']}/{EXPECTED_PAIRS} paired arrivals"
        )
    return result


def require_complete_terminal_metrics(path: Path) -> None:
    values = rows(require(path))
    expected = {(n, a, k) for n in (50, 100, 250, 1000) for a in ASPL_VALUES for k in K_VALUES}
    actual = {(int(r["n"]), int(r["target_ASPL"]), int(r["k"])) for r in values}
    incomplete = [r for r in values if int(r["checkpoint_count"]) != EXPECTED_PAIRS or int(r["paired_observed"]) != EXPECTED_PAIRS]
    if actual != expected or len(values) != len(expected) or incomplete:
        raise RuntimeError("terminal_metrics.csv must contain all 48 fully observed n × ASPL × k cells")


def load_pairs(per_run: Path):
    """Create the original random-initial plotting function's pairs mapping."""
    grouped: dict[tuple[int, int], list[tuple[list[float], list[float]]]] = defaultdict(list)
    with gzip.open(require(per_run), "rt", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        needed = {"n", "aspl", "k"} | {f"{sim}_T{i}" for sim in ("full", "mm") for i in range(10)}
        missing = sorted(needed - fields)
        if missing:
            raise RuntimeError(f"per_run.csv.gz has no required columns: {', '.join(missing)}")
        for row in reader:
            if int(row["n"]) != 1000:
                continue
            key = int(row["aspl"]), int(row["k"])
            if key not in {(a, k) for a in ASPL_VALUES for k in K_VALUES}:
                continue
            full = [float(row[f"full_T{i}"]) for i in range(10)]
            mm = [float(row[f"mm_T{i}"]) for i in range(10)]
            if not (np.all(np.isfinite(full)) and np.all(np.isfinite(mm))):
                raise RuntimeError(f"Censored/nonfinite n=1000 arrival in per_run.csv.gz for {key}")
            grouped[key].append((full, mm))
    expected = {(a, k) for a in ASPL_VALUES for k in K_VALUES}
    if set(grouped) != expected or any(len(grouped[key]) != EXPECTED_PAIRS for key in expected):
        counts = {str(key): len(grouped[key]) for key in sorted(expected)}
        raise RuntimeError(f"Expected 400 n=1000 paired records per ASPL/k cell; found {counts}")
    return dict(grouped)


def load_cdf_cells(cache: Path):
    result = {}
    for aspl in ASPL_VALUES:
        for k in K_VALUES:
            path = require(cache / f"aspl{aspl}_k{k}.npz")
            with np.load(path) as saved:
                cell = {name: saved[name].copy() for name in saved.files}
            if int(cell.get("checkpoint_count", -1)) != EXPECTED_PAIRS:
                raise RuntimeError(f"{path}: expected 400 paired trajectories")
            for key in ("full_quantiles", "mm_quantiles"):
                if cell.get(key, np.empty(0)).shape != (3, 501):
                    raise RuntimeError(f"{path}: invalid {key} shape")
            for key in ("full_arrivals", "mm_arrivals"):
                if cell.get(key, np.empty(0)).shape != (EXPECTED_PAIRS,) or not np.all(np.isfinite(cell[key])):
                    raise RuntimeError(f"{path}: censored/nonfinite final arrivals in {key}")
            result[aspl, k] = cell
    return result


def make_contact_sheet(figures: Path) -> Path:
    from PIL import Image, ImageDraw

    images = sorted(figures.glob("*.png"))
    if not images:
        raise RuntimeError("No PNG figures were written")
    thumbs = []
    for path in images:
        with Image.open(path) as original:
            original.verify()
        with Image.open(path) as original:
            image = original.convert("RGB")
            image.thumbnail((360, 240))
            thumbs.append((path.name, image.copy()))
    cols, cell_w, cell_h = 3, 380, 280
    canvas = Image.new("RGB", (cols * cell_w, ((len(thumbs) + cols - 1) // cols) * cell_h), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (name, image) in enumerate(thumbs):
        x, y = (index % cols) * cell_w, (index // cols) * cell_h
        canvas.paste(image, (x + (cell_w - image.width) // 2, y + 4))
        draw.text((x + 6, y + 248), name[:54], fill="black")
    destination = figures / "contact_sheet.png"
    canvas.save(destination)
    return destination


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(analysis: Path) -> None:
    figures = analysis / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    inputs = {
        "cell_summary": require(analysis / "cell_summary.csv"),
        "per_run": require(analysis / "per_run.csv.gz"),
        "layer_metrics": require(analysis / "layer_metrics.csv"),
        "median_layer_metrics": require(analysis / "median_layer_metrics.csv"),
        "terminal_metrics": require(analysis / "terminal_metrics.csv"),
        "cdf_cache": analysis / "trajectory_cdf_cache",
    }
    layer_rows = require_complete_layer_metrics(inputs["layer_metrics"])
    require_complete_terminal_metrics(inputs["terminal_metrics"])
    pairs = load_pairs(inputs["per_run"])
    cdf_cells = load_cdf_cells(inputs["cdf_cache"])

    random, random_path = load_original("plot_l10_aspl_chain_random_initial")
    median, median_path = load_original("plot_l10_aspl_chain_median_heatmap")
    labeled, labeled_path = load_original("plot_l10_aspl_chain_labeled_heatmap")
    size, size_path = load_original("plot_l10_aspl_chain_size_heatmap")
    trajectory, trajectory_path = load_original("plot_l10_aspl_chain_trajectory_cdf")
    clique, clique_path = load_original("plot_chain_cliques_n1000_k1_article")
    smpt, smpt_path = load_original("smpt_twoscale_paper_figures")

    # Existing publication functions, rebased to this isolated analysis output.
    smpt.OUT = figures
    summary = __import__("pandas").read_csv(inputs["cell_summary"])
    smpt.accuracy_map(summary)
    smpt.descriptor(summary)

    median.OUT = analysis
    median.FIGURES = figures
    median.ARTICLE_IMAGES = figures
    median.MEDIAN_TABLE = inputs["median_layer_metrics"]
    median.main()

    labeled.SOURCE = inputs["layer_metrics"]
    labeled.FIGURES = figures
    labeled.main()

    size.CACHE = inputs["terminal_metrics"]
    size.OUT = analysis
    size.FIGURES = figures
    size.N1000_METRICS = inputs["median_layer_metrics"]
    # CACHE is complete, so size.main() uses no checkpoint data.  Retain real
    # data roots only for its availability labels and never write to them.
    size.DATA_ROOTS = {n: ROOT / "output" / "data" / size.DATA_TEMPLATE.format(n=n) for n in size.SIZES}
    # Reuse rendering while keeping the audited CSV immutable: the legacy
    # gather() rewrites its cache even when every cached cell is complete.
    terminal = rows(inputs["terminal_metrics"])
    lookup = {(int(r["n"]), int(r["target_ASPL"]), int(r["k"])): r for r in terminal}
    shifts = {k: np.asarray([[float(lookup[n, a, k]["signed_median_shift_percent"])
                             for n in size.SIZES] for a in ASPL_VALUES]) for k in K_VALUES}
    distances = {k: np.asarray([[float(lookup[n, a, k]["relative_w1_percent"])
                                for n in size.SIZES] for a in ASPL_VALUES]) for k in K_VALUES}
    coverage = {(a, n, k): (400, 400, 0) for n in size.SIZES for a in ASPL_VALUES for k in K_VALUES}
    size.gather = lambda: (shifts, distances, coverage)
    size.main()

    trajectory.FIGURES = figures
    for k in K_VALUES:
        trajectory.draw_sheet(cdf_cells, [(a, k) for a in ASPL_VALUES], f"aspl_chain_trajectory_cdf_k{k}")
    trajectory.draw_sheet(cdf_cells, [(a, k) for a in ASPL_VALUES for k in K_VALUES], "aspl_chain_trajectory_cdf_all")

    clique.draw(cdf_cells[1, 1], figures)

    random.FIGURES = figures
    random.plot_main(layer_rows)
    random.plot_local(layer_rows)
    random.plot_completion(layer_rows)
    random.plot_final_arrivals(pairs)

    contact = make_contact_sheet(figures)
    source_paths = [random_path, median_path, labeled_path, size_path, trajectory_path, clique_path, smpt_path]
    for path in source_paths:
        target = analysis / "sources" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target) != sha256(path):
            raise RuntimeError(f"Original plotting source changed: {path}")
        shutil.copyfile(path, target)
    manifest = {
        "analysis": str(analysis),
        "inputs": {name: str(path) for name, path in inputs.items()},
        "source_scripts_sha256": {path.name: sha256(path) for path in source_paths},
        "figures": sorted(path.name for path in figures.glob("*.png")),
        "contact_sheet": contact.name,
        "validation": "400 observed paired arrivals in every ordinary-arrival cell; PNGs verified with Pillow",
    }
    (figures / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(manifest['figures'])} figures: {figures}")
    print(f"Manifest: {figures / 'figure_manifest.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, type=Path, help="Saved corrected analysis directory")
    args = parser.parse_args()
    analysis = args.analysis.resolve()
    if (analysis / "figures").exists():
        raise RuntimeError("Figures already exist; use a new analysis directory")
    sys.path.insert(0, str(ROOT))
    from extrascripts.runners.rerun_smpt_paper import log, object_hash
    sources = [Path(__file__)] + [VIS / (name + ".py") for name in (
        "plot_l10_aspl_chain_random_initial", "plot_l10_aspl_chain_median_heatmap",
        "plot_l10_aspl_chain_labeled_heatmap", "plot_l10_aspl_chain_size_heatmap",
        "plot_l10_aspl_chain_trajectory_cdf", "plot_chain_cliques_n1000_k1_article", "smpt_twoscale_paper_figures")]
    code = {str(path.relative_to(ROOT)): sha256(path) for path in sources}
    fingerprint = object_hash(code)
    inputs = {str(path.relative_to(analysis)): sha256(path) for path in
              sorted(analysis.glob("*.csv*")) + sorted((analysis / "trajectory_cdf_cache").glob("*.npz"))}
    log(analysis, "saved_figures_started", simulation=False, command=sys.argv,
        code_fingerprint=fingerprint, code_sha256=code, input_sha256=inputs,
        parameters="original chart functions; 400 paired runs per cell", seeds=None,
        trajectory_policy="saved exact arrivals; trajectory median/IQR", resume="disabled")
    try:
        render(analysis)
        for relative, sha in inputs.items():
            if sha256(analysis / relative) != sha:
                raise RuntimeError(f"Plotting altered audited input: {relative}")
        for relative, sha in code.items():
            if sha256(ROOT / relative) != sha:
                raise RuntimeError(f"Plotting source changed: {relative}")
        shutil.copyfile(Path(__file__), analysis / "sources" / Path(__file__).relative_to(ROOT))
        log(analysis, "saved_figures_completed", simulation=False, status="complete", code_fingerprint=fingerprint)
    except BaseException as error:
        log(analysis, "saved_figures_failed", simulation=False, error=repr(error), code_fingerprint=fingerprint)
        raise


if __name__ == "__main__":
    main()
