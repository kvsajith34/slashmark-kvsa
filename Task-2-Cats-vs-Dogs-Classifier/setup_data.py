"""
setup_data.py — Prepare the dataset before training.

Two modes
---------
1. Kaggle dataset  (recommended for real training)
   Download the full 25 000-image dataset from Kaggle, then run:

       python setup_data.py --source kaggle --kaggle-dir /path/to/kaggle/download

2. Sample dataset  (default — works out of the box, no Kaggle account needed)
   Downloads a small set of royalty-free cat/dog images from the web so
   you can verify the whole pipeline runs before committing to a long
   training run.

The resulting folder layout expected by ImageDataGenerator:

    data/
        train/
            cats/   *.jpg
            dogs/   *.jpg
        val/
            cats/   *.jpg
            dogs/   *.jpg
"""

import argparse
import os
import random
import shutil
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# A handful of small, public-domain images hosted on Wikimedia Commons.
# These are ONLY used in --source=sample mode so the project runs without
# any external accounts.
# ---------------------------------------------------------------------------
SAMPLE_URLS = {
    "cats": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/320px-Cat_November_2010-1a.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/b/bb/Kittyply_edit1.jpg/320px-Kittyply_edit1.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/68/Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg/320px-Orange_tabby_cat_sitting_on_fallen_leaves-Hisashi-01A.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/320px-Cat03.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Cat_poster_1.jpg/320px-Cat_poster_1.jpg",
    ],
    "dogs": [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/320px-YellowLabradorLooking_new.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/18/Dog_Breeds.jpg/320px-Dog_Breeds.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Collage_of_Nine_Dogs.jpg/320px-Collage_of_Nine_Dogs.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Beagle_puppy.jpg/320px-Beagle_puppy.jpg",
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/Taka_Shiba.jpg/320px-Taka_Shiba.jpg",
    ],
}

VAL_SPLIT = 0.2   # 20 % of images go to validation


def make_dirs(base: Path):
    for split in ("train", "val"):
        for cls in ("cats", "dogs"):
            (base / split / cls).mkdir(parents=True, exist_ok=True)


def download_samples(base: Path):
    """Download the handful of sample images defined above."""
    print("Downloading sample images …")
    for cls, urls in SAMPLE_URLS.items():
        for i, url in enumerate(urls):
            ext = url.split(".")[-1].split("?")[0] or "jpg"
            dest = base / "raw" / cls / f"{cls}_{i:04d}.{ext}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                try:
                    urllib.request.urlretrieve(url, dest)
                    print(f"  ✓ {dest.name}")
                except Exception as exc:
                    print(f"  ✗ could not fetch {url}: {exc}")
    print("Done.\n")


def split_and_copy(src_root: Path, dst_root: Path, val_split=VAL_SPLIT,
                   seed=42):
    """
    Walk src_root/<class>/ directories, shuffle each class list, then
    copy (val_split × n) images to val/ and the rest to train/.
    """
    random.seed(seed)
    for cls_dir in sorted(src_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        images = sorted(cls_dir.glob("*"))
        images = [p for p in images
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]
        random.shuffle(images)

        n_val = max(1, int(len(images) * val_split))
        splits = {"val": images[:n_val], "train": images[n_val:]}

        for split, paths in splits.items():
            for src in paths:
                dst = dst_root / split / cls / src.name
                shutil.copy2(src, dst)

        print(f"  {cls:6s} → {len(splits['train'])} train, "
              f"{len(splits['val'])} val")


def from_kaggle(kaggle_dir: str, data_dir: Path):
    """
    Copy from the flat Kaggle directory (train/*.jpg where filenames are
    like cat.0.jpg, dog.1234.jpg) into our class-labelled layout.
    """
    src = Path(kaggle_dir)
    raw = data_dir / "raw"
    (raw / "cats").mkdir(parents=True, exist_ok=True)
    (raw / "dogs").mkdir(parents=True, exist_ok=True)

    print("Organising Kaggle images …")
    for img in src.glob("*.jpg"):
        label = "cats" if img.name.startswith("cat") else "dogs"
        shutil.copy2(img, raw / label / img.name)
    print("Done.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Set up the cats-vs-dogs dataset."
    )
    parser.add_argument(
        "--source", choices=["sample", "kaggle"], default="sample",
        help="'sample' downloads a tiny demo set; 'kaggle' uses your local "
             "copy of the Kaggle dataset.",
    )
    parser.add_argument(
        "--kaggle-dir", default="",
        help="Path to the flat Kaggle train/ folder (only used with "
             "--source kaggle).",
    )
    parser.add_argument(
        "--data-dir", default="data",
        help="Root directory where the organised dataset will be written "
             "(default: ./data).",
    )
    parser.add_argument(
        "--val-split", type=float, default=VAL_SPLIT,
        help="Fraction of images to use for validation (default 0.2).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    make_dirs(data_dir)

    if args.source == "kaggle":
        if not args.kaggle_dir:
            parser.error("--kaggle-dir is required when --source=kaggle")
        from_kaggle(args.kaggle_dir, data_dir)
    else:
        download_samples(data_dir)

    print("Splitting into train / val …")
    split_and_copy(data_dir / "raw", data_dir, val_split=args.val_split)
    print("\nDataset ready. Folder layout:")
    for p in sorted(data_dir.rglob("*")):
        if p.is_dir():
            n = len(list(p.glob("*")))
            print(f"  {p.relative_to(data_dir)}/  ({n} files)")


if __name__ == "__main__":
    main()
