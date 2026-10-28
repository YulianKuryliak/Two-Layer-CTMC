#!/usr/bin/env python3
"""Write an inspectable comparison of saved old/corrected chain figures."""
import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from extrascripts.runners.rerun_smpt_paper import log, utc


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verified_old_reference(out, manifest):
    """Recompute old summaries and render references from audited old rows."""
    metric_path = out / "sources/extrascripts/analysis/smpt_twoscale_paper_metrics.py"
    metrics = load_module("verified_old_metrics", metric_path)
    old = pd.read_csv(out / "old_per_run_recomputed.csv.gz")
    rng = np.random.default_rng(metrics.SEED)
    rows = []
    for (n, aspl, k), cell in old.groupby(["n", "aspl", "k"], sort=True):
        rows.append(dict(n=n, aspl=aspl, k=k, mean_degree=cell.mean_degree.iloc[0], **metrics.summarise(cell, rng)))
    recomputed = pd.DataFrame(rows)
    cached = pd.read_csv(ROOT / "output/analysis/smpt_twoscale_paper/cell_summary.csv").sort_values(["n", "aspl", "k"]).reset_index(drop=True)
    assert len(cached) == len(recomputed) == 48
    for field in recomputed:
        assert np.allclose(cached[field].to_numpy(dtype=float), recomputed[field].to_numpy(dtype=float),
                           rtol=1e-11, atol=1e-9, equal_nan=True), f"Cached old summary mismatch: {field}"
    reference = out / "old_reference"
    reference.mkdir(exist_ok=False)
    recomputed.to_csv(reference / "cell_summary.csv", index=False)
    median_rows = []
    for (aspl, k), cell in old[old.n == 1000].groupby(["aspl", "k"], sort=True):
        for c in range(1, 10):
            f, m = cell[f"full_T{c}"].to_numpy(), cell[f"mm_T{c}"].to_numpy()
            median_rows.append(dict(target_ASPL=aspl, k=k, community=c, paired_runs=400,
                median_full=np.median(f), median_mm=np.median(m),
                signed_relative_median_shift_percent=metrics.median_shift(f, m),
                relative_w1_percent=metrics.rel_w1(f, m)))
    pd.DataFrame(median_rows).to_csv(reference / "median_layer_metrics.csv", index=False)
    figures = reference / "figures"
    figures.mkdir()
    source = out / "sources/extrascripts/visualization"
    smpt = load_module("reference_smpt", source / "smpt_twoscale_paper_figures.py")
    smpt.OUT = figures
    smpt.accuracy_map(recomputed)
    median = load_module("reference_median", source / "plot_l10_aspl_chain_median_heatmap.py")
    median.OUT = reference
    median.MEDIAN_TABLE = reference / "median_layer_metrics.csv"
    median.FIGURES = median.ARTICLE_IMAGES = figures
    median.main()
    size = load_module("reference_size", source / "plot_l10_aspl_chain_size_heatmap.py")
    lookup = {(int(row.n), int(row.aspl), int(row.k)): row for _, row in recomputed.iterrows()}
    shifts = {k: np.asarray([[lookup[n, a, k].median_shift_pct for n in size.SIZES]
                             for a in size.ASPL_VALUES]) for k in size.K_VALUES}
    distances = {k: np.asarray([[lookup[n, a, k].rel_w1_pct for n in size.SIZES]
                                for a in size.ASPL_VALUES]) for k in size.K_VALUES}
    coverage = {(a, n, k): (400, 400, 0) for n in size.SIZES for a in size.ASPL_VALUES for k in size.K_VALUES}
    size.gather = lambda: (shifts, distances, coverage)
    size.FIGURES = figures
    size.main()
    log(out, "old_reference_verified", simulation=False, paired_tasks=19200, cells=48,
        cached_summary_matches_recomputed=True, seed=metrics.SEED, N_BOOT=metrics.N_BOOT, N_PERM=metrics.N_PERM,
        rendering="same archived renderers applied to verified old rows; no stale paper PNG assumptions")
    return reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    out = args.analysis.resolve()
    destination = out / "comparison_uk.md"
    if destination.exists():
        raise RuntimeError("Comparison already exists; do not overwrite old analysis")
    code_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    log(out, "comparison_report_started", simulation=False, command=sys.argv, code_sha256=code_sha,
        parameters="48 matched 400-pair cells; descriptive changes in original metrics", seeds=None,
        trajectory_policy="saved summaries only", resume="disabled")
    try:
        snapshot = out / "sources" / Path(__file__).relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__), snapshot)
        manifest = json.loads((out / "analysis_manifest.json").read_text())
        figures = json.loads((out / "figures/figure_manifest.json").read_text())
        comparison = pd.read_csv(out / "cell_comparison.csv")
        assert manifest["status"] == "complete" and len(comparison) == 48
        assert manifest["full_rows_exact"] == 19200
        reference = verified_old_reference(out, manifest)
        delta = comparison.rel_w1_pct_new - comparison.rel_w1_pct_old
        findings = dict(created_utc=utc(), full_rows_exact=19200,
            old_rel_w1_min=float(comparison.rel_w1_pct_old.min()), old_rel_w1_max=float(comparison.rel_w1_pct_old.max()),
            new_rel_w1_min=float(comparison.rel_w1_pct_new.min()), new_rel_w1_max=float(comparison.rel_w1_pct_new.max()),
            old_rel_w1_median_across_cells=float(comparison.rel_w1_pct_old.median()),
            new_rel_w1_median_across_cells=float(comparison.rel_w1_pct_new.median()),
            w1_improved_cells=int((delta < -1e-10).sum()), w1_worsened_cells=int((delta > 1e-10).sum()),
            old_median_absolute_median_shift=float(comparison.median_shift_pct_old.abs().median()),
            new_median_absolute_median_shift=float(comparison.median_shift_pct_new.abs().median()),
            new_median_shift_min=float(comparison.median_shift_pct_new.min()),
            new_median_shift_max=float(comparison.median_shift_pct_new.max()),
            new_w1_perm_p_below_005_cells=int((comparison.rel_w1_perm_p_new < .05).sum()),
            old_w1_perm_p_below_005_cells=int((comparison.rel_w1_perm_p_old < .05).sum()),
            mm_event_sequences_changed=manifest["mm_event_sequences_changed"],
            common_30000_full_arrivals=manifest["common_30000_full_T9_observed"],
            common_30000_mm_arrivals=manifest["common_30000_mm_T9_observed"],
            png_figure_count=len([name for name in figures["figures"] if name != "contact_sheet.png"]))
        lines = ["# Порівняння попередніх і виправлених chain результатів", "",
            f"Нові дані: `{manifest['source']}`. Перевірено **19 200 пар**, 48 конфігурацій по 400 повторів.", "",
            f"Full не змінився: усі **{manifest['full_rows_exact']}** повних рядків точно збігаються зі збереженими джерелами. "
            f"Послідовності інфікування TwoScale змінилися у **{manifest['mm_event_sequences_changed']}** задачах; "
            f"послідовності прибуття — у **{manifest['mm_arrival_sequences_changed']}**.", "",
            "Новий набір має версію `chain_twoscale_residual_midpoint_v1`: це виправлення залишкового локального годинника "
            "та RNG порівняно з даними старих рисунків. Наявність оптимізованих копій коду сама по собі не доводить, що цей набір "
            "виконано ними. Цей звіт не приписує наукові зміни оптимізації швидкості.", "",
            "## Зміни метрик", "",
            "| Показник | Старі дані | Нові дані |", "|---|---:|---:|",
            f"| Медіана відносної W₁ серед 48 конфігурацій | {findings['old_rel_w1_median_across_cells']:.2f}% | {findings['new_rel_w1_median_across_cells']:.2f}% |",
            f"| Діапазон відносної W₁ | {findings['old_rel_w1_min']:.2f}–{findings['old_rel_w1_max']:.2f}% | {findings['new_rel_w1_min']:.2f}–{findings['new_rel_w1_max']:.2f}% |",
            f"| Медіана абсолютного зсуву медіанного T₉ серед конфігурацій | {findings['old_median_absolute_median_shift']:.2f}% | {findings['new_median_absolute_median_shift']:.2f}% |", "",
            f"W₁ зменшилась у **{findings['w1_improved_cells']} із 48** конфігурацій і збільшилась у **{findings['w1_worsened_cells']}**. "
            "Це описові зміни точкових оцінок; окремий тест статистичної значущості саме зміни старе→нове не виконувався.", "",
            "## Приклади для n=1000", "",
            "| ASPL | k | W₁ стара, % | W₁ нова, % | Зсув медіани старий, % | Зсув медіани новий, % |", "|---:|---:|---:|---:|---:|---:|"]
        for _, row in comparison[(comparison.n == 1000) & comparison.aspl.isin([1, 6])].sort_values(["aspl", "k"]).iterrows():
            lines.append(f"| {int(row.aspl)} | {int(row.k)} | {row.rel_w1_pct_old:.2f} | {row.rel_w1_pct_new:.2f} | {row.median_shift_pct_old:+.2f} | {row.median_shift_pct_new:+.2f} |")
        lines.extend(["", "Знак зсуву: від'ємний означає, що TwoScale досягає останньої спільноти раніше Full; додатний — пізніше.", "",
            "## Горизонти та невизначеність", "",
            f"Дві історичні траєкторії Full продовжені до 40 000, TwoScale має горизонт 30 000. "
            f"Щоб відтворити старі рисунки, використано ті самі Full-продовження; у `checkpoint_comparison.csv.gz` вони "
            f"позначені окремо. На спільному горизонті 30 000 T₉ спостерігається у **{findings['common_30000_full_arrivals']}/19 200 Full** "
            f"та **{findings['common_30000_mm_arrivals']}/19 200 TwoScale**. Порівняння завершення при різних горизонтах не робиться.", "",
            "Пермутаційні тести й network-block bootstrap відтворюють метод старого скрипта: 2 000 перестановок, "
            "2 000 bootstrap-вибірок, seed 20260930. Перевірки p<0.05 є окремими тестами по клітинках без поправки "
            "на множинність. IQR траєкторій показує розкид між реалізаціями, а не довірчий інтервал.", "",
            "## Файли", "",
            "- [Усі нові рисунки: контактний лист](figures/contact_sheet.png)",
            "- [Карта W₁ і зсуву медіани](figures/fig_accuracy_map.pdf)",
            "- [Дескриптори похибки](figures/fig_descriptor.pdf)",
            "- [Похибка прибуття по спільнотах](figures/aspl_chain_median_arrival_error_map_labeled.pdf)",
            "- [Похибка по розміру спільноти й ASPL](figures/aspl_chain_size_terminal_arrival_heatmap.pdf)",
            "- [Поширення та CDF для clique-chain](figures/chain_cliques_n1000_k1_propagation_cdf_random_initial.pdf)",
            "- [Усі зміни по 48 конфігураціях](cell_comparison.csv)",
            "- [Порівняння рисунків до й після](before_after_overview.png)",
            "- [Повна перевірка checkpoint та горизонтів](checkpoint_comparison.csv.gz)", "",
            "Старі рисунки та manuscript-файли не перезаписані. Для clique–leaf і одиночних спільнот у цьому "
            "chain-перерахунку нових даних немає; їхні рисунки залишаються попередніми. Оригінальні plotting-скрипти "
            "збережені в `sources/`; схема методу не залежить від результатів і не перегенеровувалась.", ""])
        lines.extend(["Старий підсумковий CSV незалежно перераховано з усіх перевірених історичних пар: "
                      "усі його числові поля, включно з bootstrap/пермутаційними оцінками, збігаються. "
                      "Панелі «до» в огляді заново побудовані з цих перевірених старих даних тим самим архівованим кодом; "
                      "це дозволяє уникнути залежності від можливих ручних змін PNG статті. Кожна карта має власну "
                      "позначену шкалу кольорів; точні зміни слід читати за числами, а не лише кольором.", ""])
        destination.write_text("\n".join(lines))
        (out / "comparison_findings.json").write_text(json.dumps(findings, indent=2) + "\n")
        # Visual before/after overview; quantitative tables govern comparison
        # because each original heatmap has its own explicitly labelled scale.
        chosen = [(reference / "figures" / name, name) for name in (
            "fig_accuracy_map.png", "aspl_chain_median_arrival_error_map_labeled.png",
            "aspl_chain_size_terminal_arrival_heatmap.png")]
        canvas = Image.new("RGB", (1500, 3 * 560 + 70), "white")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 25)
        draw.text((25, 15), "Previous saved results", fill="black", font=font)
        draw.text((775, 15), "Corrected residual-clock results", fill="black", font=font)
        overview_sources = []
        for row, (old_path, name) in enumerate(chosen):
            overview_sources.append({"old": str(old_path), "new": str(out / "figures" / name),
                "old_sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
                "new_sha256": hashlib.sha256((out / "figures" / name).read_bytes()).hexdigest()})
            for col, path in enumerate((old_path, out / "figures" / name)):
                with Image.open(path) as opened:
                    image = opened.convert("RGB")
                    image.thumbnail((730, 530))
                    canvas.paste(image, (col * 750 + (750 - image.width) // 2,
                                        70 + row * 560 + (530 - image.height) // 2))
        canvas.save(out / "before_after_overview.png")
        (out / "before_after_sources.json").write_text(json.dumps(overview_sources, indent=2) + "\n")
        log(out, "comparison_report_completed", simulation=False, status="complete", code_sha256=code_sha,
            report=str(destination), findings=findings)
        print(json.dumps(findings, indent=2))
    except BaseException as error:
        log(out, "comparison_report_failed", simulation=False, code_sha256=code_sha, error=repr(error))
        raise


if __name__ == "__main__":
    main()
