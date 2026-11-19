"""Download the pinned datasets in data/manifest.yaml into data/raw/ and verify hashes.

Hashes are recorded in data/checksums.json on first download (commit that file);
later downloads must match exactly.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.yaml"
CHECKSUMS = ROOT / "data" / "checksums.json"
RAW = ROOT / "data" / "raw"


def raw_path(key: str, split: str, filename: str) -> Path:
    return RAW / key / f"{split}{Path(filename).suffix}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    manifest = yaml.safe_load(MANIFEST.read_text())["datasets"]
    checksums = json.loads(CHECKSUMS.read_text()) if CHECKSUMS.exists() else {}
    with httpx.Client(follow_redirects=True, timeout=120) as http:
        for key, ds in manifest.items():
            for split, filename in ds["files"].items():
                dest = raw_path(key, split, filename)
                ident = f"{key}/{split}"
                if not dest.exists():
                    url = f"https://huggingface.co/datasets/{ds['repo']}/resolve/{ds['revision']}/{filename}"
                    print(f"downloading {ident} <- {url}")
                    r = http.get(url)
                    r.raise_for_status()
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(r.content)
                digest = sha256(dest)
                if ident in checksums and checksums[ident] != digest:
                    print(f"CHECKSUM MISMATCH for {ident}: {digest} != {checksums[ident]}")
                    return 1
                checksums[ident] = digest
                print(f"ok {ident} {digest[:12]} {dest.stat().st_size} bytes")
    CHECKSUMS.write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
