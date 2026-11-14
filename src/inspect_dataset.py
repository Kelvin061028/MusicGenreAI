"""Check that the GTZAN folder has the expected genre subfolders and audio files."""
from collections import Counter
from pathlib import Path
import sys

try:
    from .config import GENRES, RAW_DATA_DIR
except ImportError:
    from config import GENRES, RAW_DATA_DIR

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


def main() -> None:
    if not RAW_DATA_DIR.exists():
        print(f"Dataset folder not found: {RAW_DATA_DIR}")
        print("Create data/genres_original/ and place one folder per genre inside it.")
        sys.exit(1)

    counts = Counter()
    missing = []
    for genre in GENRES:
        folder = RAW_DATA_DIR / genre
        if not folder.is_dir():
            missing.append(genre)
            counts[genre] = 0
            continue
        counts[genre] = sum(
            1 for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
        )

    print("GTZAN dataset inspection")
    print("-" * 32)
    for genre in GENRES:
        print(f"{genre:12s}: {counts[genre]:4d} audio files")

    total = sum(counts.values())
    print("-" * 32)
    print(f"Total audio files: {total}")

    if missing:
        print("\nMissing genre folders:", ", ".join(missing))
    if total == 0:
        print("\nNo audio files found. Check the dataset folder structure.")
    elif total < 1_000:
        print("\nNote: fewer than 1,000 files were found. Check whether extraction is complete.")
    else:
        print("\nDataset files found. Next, run: python -m src.prepare_dataset")


if __name__ == "__main__":
    main()
