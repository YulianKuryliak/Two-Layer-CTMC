#!/usr/bin/env python3
"""ZIP64 a completed, checksum-verified release using large NTFS-friendly writes."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile


def archive_dataset(folder, output):
    assert json.loads((folder / 'full_verification.json').read_text())['status'] == 'passed'
    assert (folder / 'SHA256SUMS').is_file()
    assert not output.exists(), 'Never overwrite an existing archive'
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=4, allowZip64=True) as archive:
        for path in sorted(folder.rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts:
                continue
            name = str(Path(folder.name) / path.relative_to(folder))
            if path.suffix == '.tar':
                info = zipfile.ZipInfo.from_file(path, name)
                info.compress_type = zipfile.ZIP_STORED
                with path.open('rb') as source, archive.open(info, 'w', force_zip64=True) as target:
                    shutil.copyfileobj(source, target, length=4 * 2**20)
                print(f'Archived {path.name}: {path.stat().st_size / 2**30:.3f} GiB', flush=True)
            else:
                archive.write(path, name)
    with zipfile.ZipFile(output) as archive:
        assert archive.read(folder.name + '/SHA256SUMS') == (folder / 'SHA256SUMS').read_bytes()
        assert archive.read(folder.name + '/full_verification.json') == (folder / 'full_verification.json').read_bytes()
        assert all(info.file_size == (folder.parent / info.filename).stat().st_size for info in archive.infolist())
    print(f'Ready: {output} ({output.stat().st_size / 2**30:.3f} GiB)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    folder = args.folder.resolve()
    archive_dataset(folder, args.out or folder.with_suffix('.zip'))


if __name__ == '__main__':
    main()
