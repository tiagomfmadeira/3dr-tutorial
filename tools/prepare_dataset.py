"""Package the tutorial camera cache without executing its pickle contents."""

import argparse
import hashlib
from pathlib import Path
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    destination = root / "release"
    destination.mkdir(exist_ok=True)
    cache = args.source / "cache" / "camera_data.pickle"
    if not cache.is_file():
        raise SystemExit("Expected cache/camera_data.pickle in the source folder.")
    manifest = f"{sha256(cache)}  cache/camera_data.pickle\n"
    archive = destination / "3dr-tutorial-cache-v1.0.0.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        bundle.write(cache, "cache/camera_data.pickle")
        bundle.write(root / "docs" / "DATASET_README.md", "DATASET_README.md")
        bundle.writestr("SHA256SUMS.txt", manifest)
    # Verify every ZIP member's CRC; no pickle is deserialized.
    with zipfile.ZipFile(archive) as bundle:
        bad_member = bundle.testzip()
        if bad_member:
            raise SystemExit(f"ZIP verification failed: {bad_member}")
    checksum = destination / "SHA256SUMS.txt"
    checksum.write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
    print(f"Verified {archive} ({archive.stat().st_size:,} bytes)")
    print(f"Archive checksum: {checksum}")


if __name__ == "__main__":
    main()
