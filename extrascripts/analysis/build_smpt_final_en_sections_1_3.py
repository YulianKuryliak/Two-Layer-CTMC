"""Build only English Sections 1--3 in the original SMPT CAS format.

This is a document build and translation-parity check, not an experiment.
The editable root is generated from the maintained English section source.
"""
from pathlib import Path
import difflib
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper/SMPT paper"
FOLDER = PAPER / "final_en_sections_1_3_20261009"
NAME = "main-SMPT"
BASE = PAPER / "claude-version_merged_parallel_EN.tex"
TEMPLATE = PAPER / "main-SMPT-twoscale.tex"
UK = PAPER / "readability_revision_parallel_black_20261008/sections_1_3_UK.tex"
EN = FOLDER / "sections_1_3_EN.tex"
SNAPSHOT = FOLDER / "user_sections_1_3_UK_snapshot.tex"


def read(path):
    return path.read_text(encoding="utf-8-sig")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_math(text):
    return re.sub(r"\s+", "", text)


def commands(text, name):
    return re.findall(r"\\" + name + r"\{([^}]+)\}", text)


def run(*args):
    result = subprocess.run(args, cwd=PAPER, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(" ".join(args) + "\n" + result.stdout[-7000:] + result.stderr[-2000:])


def write_diff(old, new, destination, old_name, new_name):
    destination.write_text("".join(difflib.unified_diff(
        old.splitlines(True), new.splitlines(True), fromfile=old_name, tofile=new_name
    )), encoding="utf-8")


def main():
    assert UK.read_bytes() == SNAPSHOT.read_bytes(), (
        "The user's Ukrainian source changed after the translation snapshot; "
        "review the new edits before rebuilding."
    )
    archived = PAPER / "readability_revision_parallel_black_20261008/originals/claude-version_merged_parallel_EN.tex"
    assert BASE.read_bytes() == archived.read_bytes(), "Original English baseline changed"
    uk, en, original = read(UK), read(EN), read(BASE)
    verification_path = FOLDER / "verification.json"
    root_path = PAPER / (NAME + ".tex")
    if verification_path.exists() and root_path.exists():
        previous = json.loads(read(verification_path))
        if "root_sha256" in previous:
            assert sha(root_path) == previous["root_sha256"], (
                "The final root was edited directly. Build it with LuaLaTeX, or "
                "reconcile its edits with the English section source before regeneration."
            )

    labels = commands(uk, "label")
    assert labels == commands(en, "label"), "Translation changed label order"
    cite_pattern = r"\\cite[pt](?:\[[^\]]*\])*\{[^}]*\}"
    assert re.findall(cite_pattern, uk) == re.findall(cite_pattern, en), "Translation changed citations"
    equation_pattern = r"\\begin\{equation\}(.*?)\\end\{equation\}"
    equations_uk = re.findall(equation_pattern, uk, re.S)
    equations_en = re.findall(equation_pattern, en, re.S)
    assert list(map(normalize_math, equations_uk)) == list(map(normalize_math, equations_en)), "Numbered equations differ"
    display_pattern = r"\\\[(.*?)\\\]"
    assert list(map(normalize_math, re.findall(display_pattern, uk, re.S))) == list(map(normalize_math, re.findall(display_pattern, en, re.S))), "Display equations differ"
    tikz_pattern = r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}"
    assert re.search(tikz_pattern, uk, re.S).group() == re.search(tikz_pattern, en, re.S).group(), "TikZ figure differs"
    heading_pattern = r"\\(section|subsection|paragraph)\{"
    assert re.findall(heading_pattern, uk) == re.findall(heading_pattern, en), "Heading structure differs"
    assert len(re.findall(r"\\section\{", en)) == 3

    # Retain later-section reference numbers from the original full article.
    # Their text is excluded from this partial document; use non-linking refs.
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", en))
    later = sorted(refs - set(labels))
    aux = read(PAPER / "claude-version_merged_parallel_EN.aux")
    external_lines = ["% Reference numbers for omitted sections, from the original English article.\n"]
    external_numbers = {}
    for label in later:
        match = re.search(r"\\newlabel\{" + re.escape(label) + r"\}\{\{([^}]+)\}\{([^}]+)\}", aux)
        assert match, f"Missing external label {label}"
        number, page = match.groups()
        external_numbers[label] = number
        external_lines.append("\\newlabel{" + label + "}{{" + number + "}{" + page + "}{}{}{}}\n")
    for label in later:
        en = en.replace("\\ref{" + label + "}", "\\ref*{" + label + "}")

    # Keep the original CAS preamble, title, authors and affiliations verbatim.
    # Use the native CAS front matter with a provisional abstract and keywords.
    # Highlights and later sections are omitted.
    front = original.split(r"\begin{abstract}", 1)[0]
    template_preamble = read(TEMPLATE).split(r"\begin{document}", 1)[0]
    template_preamble = template_preamble.replace(
        r"\usepackage{graphicx}",
        r"\usepackage{graphicx}" + "\n" + r"\usepackage{tikz}" + "\n" +
        r"\usetikzlibrary{positioning,fit,arrows.meta}", 1
    )
    front = template_preamble + r"\begin{document}" + front.split(r"\begin{document}", 1)[1]
    front = front.replace(r"\begin{document}", (
        "% Later-section reference numbers are preserved for this partial manuscript.\n"
        "\\AtBeginDocument{\n" + "".join(external_lines) + "}\n"
        + r"\begin{document}"
    ), 1)
    # The Ukrainian review file used page breaks and black links for comparison.
    # The final journal document follows the twoscale template's continuous flow.
    rendered_en = "\n".join(line for line in en.splitlines() if line not in (
        r"\clearpage", r"\begingroup\color{black}", r"\endgroup",
        r"\hypersetup{citecolor=black,linkcolor=black}",
    )) + "\n"
    rendered_en = rendered_en.replace(r"\begin{algorithm}[H]", r"\begin{algorithm}[tbp]")
    rendered_en = rendered_en.replace(r"\fbox{\begin{minipage}{0.94\linewidth}", "")
    rendered_en = rendered_en.replace(r"\end{minipage}}", "")
    root = front + (
        "% Provisional abstract for checking the native SMPT front-matter layout.\n"
        "\\begin{abstract}\n"
        "Stochastic epidemic models often combine detailed contact networks within communities "
        "with aggregate information about contacts between them. We describe a two-scale "
        "simulator for this setting. Each community is simulated by a local Gillespie engine, "
        "while a macro layer generates cross-community infections from aggregate rates and "
        "places them uniformly among susceptible nodes. We specify the local event clocks, "
        "the import rule, and the grid-based scheduler. We distinguish the implemented "
        "scheduler from an ideal continuous-time process, which is exact for a surrogate "
        "network with uniform inter-community coupling. Comparisons with full-network "
        "Gillespie simulation examine how this approximation changes infection arrival "
        "times and their distributions. The aim is to identify conditions under which "
        "aggregate coupling reproduces the quantities of interest and to explain the "
        "sources of discrepancy.\n"
        "\\end{abstract}\n\n"
        "\\begin{keywords}\n"
        "Stochastic simulation \\sep Gillespie algorithm \\sep Multiscale simulation "
        "\\sep Community networks \\sep Epidemic modelling \\sep Approximation error\n"
        "\\end{keywords}\n\n"
        "\\maketitle\n"
    ) + rendered_en + (
        "\n\\bibliographystyle{cas-model2-names}\n"
        "\\bibliography{twoscale_references}\n\\end{document}\n"
    )
    root = root.replace("% !TeX root = ../main-SMPT.tex\n", "", 1)
    (PAPER / (NAME + ".tex")).write_text(root, encoding="utf-8")
    assert r"\begin{abstract}" in root and r"\begin{keywords}" in root
    assert r"\input{" not in root, "The final manuscript must not depend on auxiliary TEX files"
    assert r"\clearpage" not in root, "Review-document page breaks must not enter the final article"
    title_pattern = r"\\title\[mode=title\]\{[^\n]+"
    assert re.search(title_pattern, root).group() == re.search(title_pattern, original).group()

    old_start = original.index(r"\section{Introduction}")
    old_end = original.index(r"\section{What the aggregate interface preserves")
    write_diff(original[old_start:old_end], read(EN), FOLDER / "english_vs_original.diff", str(BASE.relative_to(ROOT)), str(EN.relative_to(ROOT)))
    previous_en = PAPER / "readability_revision_parallel_black_20261008/sections_1_3_EN.tex"
    write_diff(read(previous_en), read(EN), FOLDER / "english_vs_previous.diff", str(previous_en.relative_to(ROOT)), str(EN.relative_to(ROOT)))

    run("pdflatex", "-recorder", "-interaction=nonstopmode", "-halt-on-error", NAME + ".tex")
    run("bibtex", NAME)
    run("pdflatex", "-recorder", "-interaction=nonstopmode", "-halt-on-error", NAME + ".tex")
    run("pdflatex", "-recorder", "-interaction=nonstopmode", "-halt-on-error", NAME + ".tex")
    run("pdftotext", "-layout", NAME + ".pdf", "final_en_sections_1_3_20261009/document.txt")
    log = read(PAPER / (NAME + ".log"))
    template_log = read(PAPER / "main-SMPT-twoscale.log")
    template_box_signatures = {
        line.split(" detected at line ")[0] for line in template_log.splitlines()
        if line.startswith("Overfull")
    }
    title_line = next(i for i, line in enumerate(root.splitlines(), 1) if line == r"\maketitle")
    inherited_boxes = [line for line in log.splitlines() if (
        line.startswith("Overfull") and
        line.split(" detected at line ")[0] in template_box_signatures and
        line.endswith(" detected at line " + str(title_line))
    )]
    assert len(inherited_boxes) <= 1, "Unexpected repeated title-box overflow"
    blockers = [line for line in log.splitlines() if (
        "undefined" in line.lower() or "Missing character" in line or
        ("Overfull" in line and line not in inherited_boxes) or "multiply defined" in line.lower() or
        "destination with the same identifier" in line
    )]
    assert not blockers, "LaTeX verification failed:\n" + "\n".join(blockers)
    text = read(FOLDER / "document.txt")
    assert "Stochastic epidemic models often combine" in text
    assert "Approximation error" in text
    assert "1. Introduction" in text
    assert "2. Related work" in text
    assert "3. The two-scale simulation method" in text
    assert "4. What the aggregate interface" not in text
    fls = read(PAPER / (NAME + ".fls"))
    for filename in ("cas-sc.cls", "cas-common.sty"):
        assert "INPUT ./" + filename in fls, f"Local template file was not loaded: {filename}"
    blg = read(PAPER / (NAME + ".blg"))
    assert "The style file: cas-model2-names.bst" in blg
    assert "Database file #1: twoscale_references.bib" in blg
    geometry_pattern = r"^\* \\(?:paperwidth|paperheight|textwidth|textheight|oddsidemargin|evensidemargin|topmargin|headheight|headsep|footskip|columnsep)=[^\n]+"
    assert re.findall(geometry_pattern, log, re.M) == re.findall(geometry_pattern, template_log, re.M), "Page geometry differs from twoscale"

    verification = {
        "scope": "Original title/author front matter, provisional abstract, six keywords and English Sections 1--3",
        "abstract_status": "provisional draft for template layout",
        "original_english": {"path": str(BASE.relative_to(ROOT)), "sha256": sha(BASE)},
        "user_ukrainian": {"path": str(UK.relative_to(ROOT)), "sha256": sha(UK)},
        "english_sections_sha256": sha(EN),
        "root_sha256": sha(root_path),
        "title_unchanged": True,
        "class": "cas-sc (original SMPT format)",
        "formatting_template": {"path": str(TEMPLATE.relative_to(ROOT)), "sha256": sha(TEMPLATE)},
        "twoscale_page_geometry_matches": True,
        "continuous_section_flow": True,
        "template_link_colours": True,
        "template_algorithm_float_placement": "tbp",
        "local_template_files": {name: sha(PAPER / name) for name in (
            "cas-sc.cls", "cas-common.sty", "cas-model2-names.bst", "twoscale_references.bib"
        )},
        "local_template_loading_verified": True,
        "same_heading_structure": True,
        "same_label_sequence": True,
        "same_citation_sequence": True,
        "numbered_equations": len(equations_en),
        "same_display_equations": True,
        "tikz_identical": True,
        "references_to_omitted_sections": external_numbers,
        "latex_blockers": blockers,
        "inherited_template_title_box_warnings": inherited_boxes,
        "pdf_sha256": sha(PAPER / (NAME + ".pdf")),
        "build": "pdfLaTeX with recorder, BibTeX, pdfLaTeX twice; pdftotext; local template inputs checked",
        "simulations_run": False,
        "translation_policy": "Sections 1--3 mirror saved Ukrainian content; provisional abstract added at user request",
    }
    verification_path.write_text(json.dumps(verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"pdf": NAME + ".pdf", "numbered_equations": len(equations_en), "labels": len(labels), "verification": "passed"}))


if __name__ == "__main__":
    main()
