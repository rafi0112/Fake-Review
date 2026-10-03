"""
download_dataset.py

Downloads the AiGen-FoodReview dataset (Gambetti & Han, ICWSM 2024) from its
official Zenodo record and lays it out as:

    data/
      train.csv
      val.csv
      test.csv
      images/
        <ID>.jpg  (one file per review, ~20,144 images)

Zenodo record: https://zenodo.org/records/10511456  (MIT license)

Run:
    python download_dataset.py --out data
"""

import argparse
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_RECORD = "10511456"
FILES = {
    "train.csv": "train.csv",
    "val.csv": "val.csv",
    "test.csv": "test.csv",
    "metadata.txt": "metadata.txt",
    "images.zip": "images.zip",
}
BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    if dest.exists():
        print(f"  already have {dest.name}, skipping")
        return
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as pbar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                pbar.update(len(chunk))
        tmp.rename(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data", help="output directory")
    ap.add_argument(
        "--skip-images",
        action="store_true",
        help="download CSVs only (images.zip is ~1.3GB); use if you already have images/",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading AiGen-FoodReview into {out_dir.resolve()}")
    for local_name, remote_name in FILES.items():
        if local_name == "images.zip" and args.skip_images:
            continue
        url = f"{BASE_URL}/{remote_name}?download=1"
        dest = out_dir / local_name
        print(f"-> {local_name}")
        try:
            download_file(url, dest)
        except requests.exceptions.RequestException as e:
            print(f"FAILED to download {local_name}: {e}", file=sys.stderr)
            print(
                "If this keeps failing, download manually from "
                "https://zenodo.org/records/10511456 and place the files in "
                f"{out_dir.resolve()}",
                file=sys.stderr,
            )
            sys.exit(1)

    images_zip = out_dir / "images.zip"
    images_dir = out_dir / "images"
    if images_zip.exists() and not images_dir.exists():
        print("Unzipping images.zip ...")
        with zipfile.ZipFile(images_zip) as zf:
            zf.extractall(out_dir)
        print(f"Images extracted to {images_dir.resolve()}")

    print("\nDone. Expected layout:")
    print(f"  {out_dir}/train.csv")
    print(f"  {out_dir}/val.csv")
    print(f"  {out_dir}/test.csv")
    print(f"  {out_dir}/images/<ID>.jpg")
    print(
        "\nOpen train.csv once and confirm the column names for the id, "
        "review text and label fields -- pass them to extract_features.py "
        "via --id-col / --text-col / --label-col if they differ from the "
        "defaults (id, text, label)."
    )


if __name__ == "__main__":
    main()
