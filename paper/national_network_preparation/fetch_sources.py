"""Download official INSEE inputs, preserving bytes and provenance. No simulation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
URLS = {
    "mobility_2022.zip": "https://www.insee.fr/fr/statistiques/fichier/8582949/base-flux-mobilite-domicile-lieu-travail-2022_csv.zip",
    "v_commune_2025.csv": "https://www.insee.fr/fr/statistiques/fichier/8377162/v_commune_2025.csv",
    "v_departement_2025.csv": "https://www.insee.fr/fr/statistiques/fichier/8377162/v_departement_2025.csv",
    "mobility_method.pdf": "https://www.insee.fr/fr/statistiques/fichier/8582949/fiche-depl-dom-travail.pdf",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    raw = HERE / "raw"
    raw.mkdir(exist_ok=True)
    manifest = HERE / "sources.json"
    if manifest.exists():
        old = json.loads(manifest.read_text())
        for item in old["files"]:
            if digest(HERE / item["path"]) != item["sha256"]:
                raise SystemExit("Existing raw source differs; preserve it and use a new folder")
        print("Source hashes verified; downloads skipped")
        return
    win_curl = Path("/mnt/c/Windows/System32/curl.exe")
    curl = str(win_curl) if win_curl.exists() else shutil.which("curl")
    if curl is None:
        raise SystemExit("curl required")
    records = []
    for name, url in URLS.items():
        path = raw / name
        if path.exists():
            raise SystemExit(f"Unmanifested input already exists: {path}")
        output = subprocess.check_output(["wslpath", "-w", str(path)], text=True).strip() if win_curl.exists() else str(path)
        command = [curl, "--fail", "--location", "--silent", "--show-error", "--max-time", "120"]
        if win_curl.exists():
            command += ["--ssl-revoke-best-effort"]
        subprocess.run(command + [url, "--output", output], check=True)
        records.append(dict(path=str(path.relative_to(HERE)), url=url, sha256=digest(path),
                            bytes=path.stat().st_size, retrieved_utc=datetime.now(timezone.utc).isoformat()))
        print(f"Downloaded {name}: {path.stat().st_size} bytes", flush=True)
    main_paper = ROOT / "paper/SMPT paper/main-SMPT.tex"
    result = dict(
        attribution="Source: Insee. Mobility 2022 release 2025-06-26; COG geography 2025-01-01, page updated 2025-03-04.",
        mobility_page="https://www.insee.fr/fr/statistiques/8582949",
        geography_page="https://www.insee.fr/fr/information/8377162",
        license="Licence Ouverte / Open Licence 2.0 (unless otherwise specified by INSEE)",
        license_source="https://www.insee.fr/fr/information/2008466",
        files=records, main_paper=dict(path=str(main_paper.relative_to(ROOT)), sha256=digest(main_paper)),
        download_note="Certificate-chain verification retained; Windows Schannel best-effort revocation used in WSL when system Linux trust cannot validate enterprise TLS interception.",
    )
    manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
