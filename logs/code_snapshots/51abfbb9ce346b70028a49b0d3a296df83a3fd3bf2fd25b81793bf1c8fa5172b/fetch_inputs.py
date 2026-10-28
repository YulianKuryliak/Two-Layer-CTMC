"""Fetch two official design inputs; resume partial downloads, never simulate."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
URLS = {
    "population_2022_geography2025.zip": "https://www.insee.fr/fr/statistiques/fichier/8581696/base-cc-evol-struct-pop-2022_csv.zip",
    "urban_units_2025.zip": "https://www.insee.fr/fr/statistiques/fichier/4802589/UU2020_au_01-01-2025.zip",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    raw = HERE / "raw"
    raw.mkdir(exist_ok=True)
    manifest_path = HERE / "sources.json"
    if manifest_path.exists():
        for item in json.loads(manifest_path.read_text())["downloaded"]:
            assert digest(HERE / item["path"]) == item["sha256"]
        print("Verified source archives; downloads skipped")
        return
    win = Path("/mnt/c/Windows/System32/curl.exe")
    curl = str(win) if win.exists() else shutil.which("curl")
    if curl is None:
        raise SystemExit("curl required")
    records = []
    for name, url in URLS.items():
        path = raw / name
        output = subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip() if win.exists() else str(path)
        if not zipfile.is_zipfile(path):
            for attempt in range(1, 6):
                command = [curl, "--fail", "--location", "--silent", "--show-error", "--max-time", "300"]
                if win.exists():
                    command += ["--ssl-revoke-best-effort"]
                before = path.stat().st_size if path.exists() else 0
                if before:
                    command += ["--continue-at", "-"]
                result = subprocess.run(command + [url, "--output", output])
                with (HERE / "downloads.jsonl").open("a") as h:
                    h.write(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(), url=url,
                        attempt=attempt, previous_bytes=before, resume=bool(before),
                        returncode=result.returncode, complete_zip=zipfile.is_zipfile(path),
                        kind="public_source_download", simulations_run=False)) + "\n")
                if zipfile.is_zipfile(path):
                    break
                if result.returncode == 33 and before:
                    partial = path.with_name(path.name + ".partial_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
                    path.rename(partial)
                    print("Server does not support Range; partial bytes preserved, retrying full download", flush=True)
                    continue
                if result.returncode not in (0, 28):
                    raise SystemExit("Download failed; partial source preserved")
            else:
                raise SystemExit("Download incomplete; rerun to resume preserved partial bytes")
        records.append(dict(path=str(path.relative_to(HERE)), url=url, bytes=path.stat().st_size,
                            sha256=digest(path), validated_utc=datetime.now(timezone.utc).isoformat()))
        print(f"Validated {name}: {path.stat().st_size} bytes", flush=True)
    population_source = ROOT / "paper/national_network_preparation/prepared"
    main_paper = ROOT / "paper/SMPT paper/main-SMPT.tex"
    result = dict(attribution="Source: Insee; population 2022 and urban-unit membership in COG geography 2025.",
        downloaded=records,
        reused=[dict(path=str(p.relative_to(ROOT)), sha256=digest(p)) for p in
                [population_source/"F_domestic.npz", population_source/"nodes.csv", population_source/"manifest.json"]],
        main_paper=dict(path=str(main_paper.relative_to(ROOT)), sha256=digest(main_paper)),
        pages=dict(population="https://www.insee.fr/fr/statistiques/8581696",
                   urban_units="https://www.insee.fr/fr/information/4802589",
                   households="https://www.insee.fr/fr/statistiques/8647008",
                   commuting="https://www.insee.fr/fr/statistiques/8582949",
                   contact_survey="https://doi.org/10.1371/journal.pone.0133203"),
        license="INSEE Open Licence 2.0 unless otherwise stated; https://www.insee.fr/fr/information/2008466",
        downloaded_vs_catalogued="Only population and urban-unit archives downloaded here; commuting reused. Household/contact-survey sources are design references, not yet fitted inputs.",
        note="A partial population download timed out before this resumable fetcher; unvalidated bytes were preserved and resumed. No synthetic persons or epidemic trajectories generated.")
    manifest_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
