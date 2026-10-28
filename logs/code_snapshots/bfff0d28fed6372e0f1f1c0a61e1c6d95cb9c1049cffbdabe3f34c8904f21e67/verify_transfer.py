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
    text = text.replace(r"\input{appendices}", "")
    # One multi-panel plot is slightly shorter to fit the IEEE float page.
    text = re.sub(r"height=4\.(?:25|0)cm", "height=IEEE-adjusted", text)
    return re.sub(r"\s+", " ", text).strip()


def separate_floats(text):
    """Compare text and float contents independently of their queue positions."""
    floats = {}
    pattern = r"\\begin\{(figure|table)\*?\}.*?\\end\{\1\*?\}"
    def extract(match):
        label = re.search(r"\\label\{([^}]+)\}", match[0])[1]
        assert label not in floats, label
        floats[label] = normalise_body(r"\section{Introduction}" + match[0] + r"\bibliographystyle")
        return ""
    remainder = re.sub(pattern, extract, text, flags=re.S)
    return normalise_body(remainder), floats


def check_appendices():
    provenance = json.loads((HERE / "layout_manifest.json").read_text())
    original = HERE / provenance["recovered_appendix_source"]
    assert sha(original) == provenance["recovered_appendix_sha256"]
    expected = original.read_text().replace(r"\appendix", r"\appendices", 1)
    expected = expected.replace(r"\citep{", r"\cite{")
    expected = expected.replace(r"\qquad" + "\n" + r"\mathbb{E}[T_B]",
        r"\end{equation}" + "\nFor one bridge this simplifies to\n" +
        r"\begin{equation}" + "\n" + r"\mathbb{E}[T_B]", 1)
    expected = expected.replace(r"\begin{table}[tbp]", r"\begin{table}[!htbp]")
    expected = expected.replace(r"\begin{tabular}{rrrrr}", r"\setlength{\tabcolsep}{3pt}" + "\n" + r"\begin{tabular}{rrrrr}")
    expected = expected.replace(
        r"$n$ & $k$ & $\mathbb{E}[T_B]$, Eq.~\eqref{eq:clique_TB} & \Full{} mean $T_B$ & \Full{} wait after $T_B$\\",
        r"$n$ & $k$ & Exact $\mathbb{E}[T_B]$ & \Full{} $T_B$ & \Full{} wait\\")
    actual = (HERE / "appendices.tex").read_text()
    clean = lambda s: re.sub(r"\s+", " ", uncomment(s)).strip()
    assert clean(expected) == clean(actual), "Recovered appendix content changed"
    assert sha(HERE / "stfloats.sty") == provenance["stfloats_sha256"]
    return actual


def main():
    manifests = sorted(HERE.glob("migration_*/source_manifest.json"))
    assert len(manifests) == 1, "Choose the migration baseline explicitly if another is added."
    manifest_path = manifests[0]
    manifest = json.loads(manifest_path.read_text())
    source_file = manifest_path.parent / "source_main-SMPT.tex"
    source, target = source_file.read_text(), (HERE / "main.tex").read_text()
    assert sha(source_file) == manifest["source_sha256"]
    acknowledgment_changed = (HERE / "acknowledgment_change.json").is_file()
    if acknowledgment_changed:
        change = json.loads((HERE / "acknowledgment_change.json").read_text())
        before = HERE / change["backup"] / "main.tex"
        assert sha(before) == change["before_main_sha256"]
        assert before.read_text().count(change["old"]) == 1
        assert before.read_text().replace(change["old"], change["new"], 1) == target
        assert source.count(change["old"]) == 1
        source = source.replace(change["old"], change["new"], 1)
    assert separate_floats(source) == separate_floats(target), "Scientific body or figure contents changed"
    assert citation_keys(source) == citation_keys(target), "Citation keys/order changed"
    appendix = check_appendices() if (HERE / "layout_manifest.json").is_file() else ""
    labels = lambda s: re.findall(r"\\label\{([^}]+)\}", uncomment(s))
    assert collections.Counter(labels(source)) == collections.Counter(labels(target))
    assert len(labels(target + appendix)) == len(set(labels(target + appendix)))
    assert sha(HERE / "references_fixed.bib") == manifest["bibliography_sha256"]
    for name, expected in manifest["plot_data_sha256"].items():
        assert sha(HERE / name) == expected, name
    for name, expected in manifest["vendor_tex_sha256"].items():
        assert sha(HERE / name) == expected, name
    for name, expected in manifest["original_template_sha256"].items():
        assert sha(HERE / name) == expected, name
    cited = {key for group in citation_keys(target + appendix) for key in group.split(",")}
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
        "body_preserved_except_citation_layout_and_approved_acknowledgment": True,
        "approved_acknowledgment_replacement": acknowledgment_changed,
        "labels": len(labels(target + appendix)), "citation_commands": len(citation_keys(target + appendix)),
        "unique_cited_keys": len(cited), "numbered_equations": (target + appendix).count(r"\begin{equation}"),
        "figures": target.count(r"\begin{figure*}"), "tables": len(re.findall(r"\\begin\{table\*?\}", target + appendix)),
        "algorithms": target.count(r"\begin{algorithm}"), "abstract_words": words,
        "plot_tables_sha256_match": len(manifest["plot_data_sha256"]),
        "original_template_files_unchanged": len(manifest["original_template_sha256"]),
        "pages": pages, "latex_warnings_checked": "no undefined refs/cites, missing glyphs, font substitutions or overfull boxes",
        "pdf_sha256": sha(HERE / "main.pdf"), "main_tex_sha256": sha(HERE / "main.tex"),
        "appendices_restored": ["A", "B", "C"] if appendix else [],
        "limitations": (["Data availability placeholder and missing author biographies remain; other declarations were not edited."]
                        if appendix else manifest["limitations"]), "simulations_run": False,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
