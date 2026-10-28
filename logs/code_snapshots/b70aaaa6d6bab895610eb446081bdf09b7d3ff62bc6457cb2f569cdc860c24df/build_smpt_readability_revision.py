"""Build the red, bilingual rewrite of SMPT sections 1--3; no simulations."""
from pathlib import Path
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper/SMPT paper"
PARTS = PAPER / "readability_revision_20261008"
PREFIX = "claude-version_merged_plain_sections_1_3_red"


def citations(text):
    return re.findall(r"\\cite[pt](?:\[[^\]]*\])*\{[^}]*\}", text)


def run(*args):
    result = subprocess.run(args, cwd=PAPER, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(" ".join(args) + "\n" + result.stdout[-6000:] + result.stderr[-2000:])


def main():
    manifest = json.loads((PARTS / "originals_manifest.json").read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((PAPER / name).read_bytes()).hexdigest() == expected, name
    en = (PAPER / "claude-version_merged_EN.tex").read_text()
    uk_front = (PAPER / "claude-version_merged_uk_front.tex").read_text()
    en_start = en.index("\\section{Introduction}")
    en_end = en.index("\\section{What the aggregate interface preserves")
    uk_start = uk_front.index("\\section{Вступ}")
    revised = {}
    for language, old in [("EN", en[en_start:en_end]), ("UK", uk_front[uk_start:])]:
        text = (PARTS / f"sections_1_3_{language}.tex").read_text()
        assert citations(text) == citations(old), f"Citation change: {language}"
        assert re.findall(r"\\label\{([^}]+)\}", text) == re.findall(r"\\label\{([^}]+)\}", old), f"Label change: {language}"
        assert text.count("\\begin{equation}") == old.count("\\begin{equation}") == 6
        assert "\\begingroup\\color{red}" in text
        revised[language] = text
    en_new = en[:en_start] + revised["EN"] + "\n\\clearpage\n" + en[en_end:]
    (PAPER / f"{PREFIX}_EN.tex").write_text(en_new)
    front_new = uk_front[:uk_start] + revised["UK"] + "\n\\clearpage\n"
    (PARTS / "uk_front.tex").write_text(front_new)
    wrapper = (PAPER / "claude-version_merged_UK.tex").read_text()
    wrapper = wrapper.replace("\\input{claude-version_merged_uk_front.tex}", "\\input{readability_revision_20261008/uk_front.tex}")
    (PAPER / f"{PREFIX}_UK.tex").write_text(wrapper)
    for language in ("EN", "UK"):
        stem = f"{PREFIX}_{language}"
        # LuaLaTeX handles Ukrainian; the English source retains its original fonts.
        run("lualatex", "-interaction=nonstopmode", "-halt-on-error", stem + ".tex")
        run("bibtex", stem)
        run("lualatex", "-interaction=nonstopmode", "-halt-on-error", stem + ".tex")
        run("lualatex", "-interaction=nonstopmode", "-halt-on-error", stem + ".tex")
        log = (PAPER / (stem + ".log")).read_text()
        if "Label(s) may have changed" in log:
            run("lualatex", "-interaction=nonstopmode", "-halt-on-error", stem + ".tex")
    # Align the starts of the rewritten sections, allowing blank padding when
    # a translation needs an extra page. The unchanged tail starts together.
    groups = {}
    for language in ("EN", "UK"):
        stem = f"{PREFIX}_{language}"
        aux = (PAPER / (stem + ".aux")).read_text()
        starts = [1]
        for label in ("sec:related", "sec:method", "sec:analysis"):
            match = re.search(r"\\newlabel\{" + label + r"\}\{\{[^}]*\}\{(\d+)\}", aux)
            assert match, label
            starts.append(int(match.group(1)) + 1)  # unnumbered Highlights page
        raw = subprocess.check_output(["pdftotext", "-raw", str(PAPER / (stem + ".pdf")), "-"], text=True)
        count = len(raw.split("\f")) - 1
        starts.append(count + 1)
        groups[language] = [list(range(a, b)) for a, b in zip(starts, starts[1:])]
    blank = PARTS / "blank.tex"
    blank.write_text("\\documentclass[a4paper]{article}\n\\pagestyle{empty}\n\\begin{document}\n\\null\\newpage\n\\end{document}\n")
    run("lualatex", "-interaction=nonstopmode", "-halt-on-error", "-output-directory=readability_revision_20261008", "readability_revision_20261008/blank.tex")
    lines = ["\\documentclass[a3paper,landscape]{article}", "\\usepackage[a3paper,landscape,margin=0mm]{geometry}", "\\usepackage{pdfpages}", "\\pagestyle{empty}", "\\begin{document}"]
    pairs = []
    for en_pages, uk_pages in zip(groups["EN"], groups["UK"]):
        for i in range(max(len(en_pages), len(uk_pages))):
            pair = []
            for language, pages in [("EN", en_pages), ("UK", uk_pages)]:
                pair.extend([f"{PREFIX}_{language}.pdf", str(pages[i])] if i < len(pages) else ["readability_revision_20261008/blank.pdf", "1"])
            lines.append("\\includepdfmerge[nup=2x1,delta=0 0]{" + ",".join(pair) + "}")
            pairs.append(pair)
    lines.append("\\end{document}")
    merged = "claude-version_merged_bilingual_UK_EN_plain_sections_1_3_red"
    (PAPER / (merged + ".tex")).write_text("\n".join(lines) + "\n")
    run("lualatex", "-interaction=nonstopmode", "-halt-on-error", merged + ".tex")
    (PARTS / "build_manifest.json").write_text(json.dumps({"source": "claude-version_merged_EN.tex / claude-version_merged_uk_front.tex", "scope": "sections 1--3 only", "citations_and_labels": "unchanged, checked in both languages", "originals_sha256": manifest, "page_groups": groups, "bilingual_page_pairs": pairs}, ensure_ascii=False, indent=2) + "\n")
    print("Built", PAPER / (merged + ".pdf"))


if __name__ == "__main__":
    main()
