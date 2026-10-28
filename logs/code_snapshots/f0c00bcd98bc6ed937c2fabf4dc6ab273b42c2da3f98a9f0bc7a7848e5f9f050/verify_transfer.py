"""Verify the IEEE formatting transfer against its saved source; no simulations."""
from pathlib import Path
import collections
import hashlib
import json
import re
import subprocess

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def uncomment(text):
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def citation_keys(text):
    return re.findall(r"\\cite[pt]?\{([^}]+)\}", uncomment(text))


def bib_entries(path):
    result = {}
    for block in re.split(r"(?m)(?=^@)", path.read_text()):
        match = re.match(r"@\w+\s*\{\s*([^,\s]+)", block)
        if match:
            result[match[1]] = block.strip()
    return result


def body(text):
    return text[text.index(r"\section{Introduction}"):text.index(r"\bibliographystyle")]


def normalise_body(text):
    text = uncomment(body(text))
    text = text.replace(r"\citep{", r"\cite{")
    for key, author in {
        "anderson2007": "Anderson", "kuryliak2025simulating": "Kuryliak et al.",
        "ajelli2010": "Ajelli et al.",
    }.items():
        text = text.replace(r"\citet{" + key + "}", author + r"~\cite{" + key + "}")
    text = re.sub(r"\\begin\{(figure|table)\*?\}\[[^]]*\]", r"\\begin{\1}", text)
    text = re.sub(r"\\end\{(figure|table)\*\}", r"\\end{\1}", text)
    text = text.replace("Section 5.3", r"Section~\ref{sec:descriptor}")
    text = text.replace(r"\clearpage", "").replace(r"\!", "")
    # One multi-panel plot is slightly shorter to fit the IEEE float page.
    text = re.sub(r"height=4\.(?:25|0)cm", "height=IEEE-adjusted", text)
    return re.sub(r"\s+", " ", text).strip()


def main():
    manifests = sorted(HERE.glob("migration_*/source_manifest.json"))
    assert len(manifests) == 1, "Choose the migration baseline explicitly if another is added."
    manifest_path = manifests[0]
    manifest = json.loads(manifest_path.read_text())
    source_file = manifest_path.parent / "source_main-SMPT.tex"
    source, target = source_file.read_text(), (HERE / "main.tex").read_text()
    assert sha(source_file) == manifest["source_sha256"]
    assert normalise_body(source) == normalise_body(target), "Scientific body changed"
    assert citation_keys(source) == citation_keys(target), "Citation keys/order changed"
    labels = lambda s: re.findall(r"\\label\{([^}]+)\}", uncomment(s))
    assert labels(source) == labels(target)
    assert len(labels(target)) == len(set(labels(target)))
    assert sha(HERE / "references_fixed.bib") == manifest["bibliography_sha256"]
    for name, expected in manifest["plot_data_sha256"].items():
        assert sha(HERE / name) == expected, name
    for name, expected in manifest["vendor_tex_sha256"].items():
        assert sha(HERE / name) == expected, name
    for name, expected in manifest["original_template_sha256"].items():
        assert sha(HERE / name) == expected, name
    cited = {key for group in citation_keys(target) for key in group.split(",")}
    fixed = bib_entries(HERE / "references_fixed.bib")
    extra = bib_entries(HERE / "references_missing.bib")
    assert not (fixed.keys() & extra.keys())
    assert cited <= fixed.keys() | extra.keys()
    supplement_source = bib_entries(manifest_path.parent / "source_twoscale_references.bib")
    assert all(value == supplement_source[key] for key, value in extra.items())
    abstract = lambda s: re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", s, re.S)[1]
    words = len(abstract(target).split())
    assert 150 <= words <= 250
    numbers = lambda s: collections.Counter(re.findall(r"\d+(?:\.\d+)?", s))
    assert numbers(abstract(source)) == numbers(abstract(target))
    assert len(abstract(target).strip().splitlines()) == 1
    log = (HERE / "main.log").read_text()
    for marker in ("undefined", "Missing character", "LaTeX Error", "Fatal error",
                   "Label(s) may have changed", "multiply defined", "Overfull",
                   "Incompatible color", "LaTeX Font Warning"):
        assert marker not in log, marker
    assert (HERE / "main.pdf").is_file()
    fls = (HERE / "main.fls").read_text()
    for name in ("./ieeeaccess.cls", "./IEEEtran.cls"):
        assert "INPUT " + name in fls, name
    info = subprocess.check_output(["pdfinfo", str(HERE / "main.pdf")], text=True)
    pages = int(re.search(r"Pages:\s+(\d+)", info)[1])
    print(json.dumps({
        "body_preserved_except_citation_and_layout_syntax": True,
        "labels": len(labels(target)), "citation_commands": len(citation_keys(target)),
        "unique_cited_keys": len(cited), "numbered_equations": target.count(r"\begin{equation}"),
        "figures": target.count(r"\begin{figure*}"), "tables": target.count(r"\begin{table*}"),
        "algorithms": target.count(r"\begin{algorithm}"), "abstract_words": words,
        "plot_tables_sha256_match": len(manifest["plot_data_sha256"]),
        "original_template_files_unchanged": len(manifest["original_template_sha256"]),
        "pages": pages, "latex_warnings_checked": "no undefined refs/cites, missing glyphs, font substitutions or overfull boxes",
        "pdf_sha256": sha(HERE / "main.pdf"), "main_tex_sha256": sha(HERE / "main.tex"),
        "limitations": manifest["limitations"], "simulations_run": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
