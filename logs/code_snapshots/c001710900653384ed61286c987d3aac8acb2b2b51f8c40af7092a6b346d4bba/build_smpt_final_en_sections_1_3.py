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
    # Abstract, highlights, keywords and later sections are intentionally omitted.
    front = original.split(r"\begin{abstract}", 1)[0]
    front = front.replace(r"\begin{document}", (
        "% Suppress the CAS abstract heading in this sections-only document.\n"
        r"\RenewDocumentEnvironment{Abstract}{o}{}{}" "\n"
        "% Later-section reference numbers are preserved for this partial manuscript.\n"
        "\\AtBeginDocument{\n" + "".join(external_lines) + "}\n"
        + r"\begin{document}"
    ), 1)
    root = front + (
        "% Use the native CAS title and author formatting, omitting empty abstract rules.\n"
        "\\let\\SMPTsavedDashrule\\dashrule\n"
        "\\RenewDocumentCommand{\\dashrule}{O{.4pt}mm}{}\n"
        "\\maketitle\n"
        "\\let\\dashrule\\SMPTsavedDashrule\n\n"
    ) + en + (
        "\n\\bibliographystyle{cas-model2-names}\n"
        "\\bibliography{twoscale_references}\n\\end{document}\n"
    )
    root = root.replace("% !TeX root = ../main-SMPT.tex\n", "", 1)
    (PAPER / (NAME + ".tex")).write_text(root, encoding="utf-8")
    assert r"\begin{abstract}" not in root
    assert r"\input{" not in root, "The final manuscript must not depend on auxiliary TEX files"
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
    blockers = [line for line in log.splitlines() if (
        "undefined" in line.lower() or "Missing character" in line or
        "Overfull" in line or "multiply defined" in line.lower() or
        "destination with the same identifier" in line
    )]
    assert not blockers, "LaTeX verification failed:\n" + "\n".join(blockers)
    text = read(FOLDER / "document.txt")
    assert "Abstract" not in text
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

    verification = {
        "scope": "Title/author front matter and English Sections 1--3 only; no abstract",
        "original_english": {"path": str(BASE.relative_to(ROOT)), "sha256": sha(BASE)},
        "user_ukrainian": {"path": str(UK.relative_to(ROOT)), "sha256": sha(UK)},
        "english_sections_sha256": sha(EN),
        "root_sha256": sha(root_path),
        "title_unchanged": True,
        "class": "cas-sc (original SMPT format)",
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
        "pdf_sha256": sha(PAPER / (NAME + ".pdf")),
        "build": "pdfLaTeX with recorder, BibTeX, pdfLaTeX twice; pdftotext; local template inputs checked",
        "simulations_run": False,
        "translation_policy": "Mirror saved Ukrainian content; no independent scientific additions or changes",
    }
    verification_path.write_text(json.dumps(verification, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"pdf": NAME + ".pdf", "numbered_equations": len(equations_en), "labels": len(labels), "verification": "passed"}))


if __name__ == "__main__":
    main()
