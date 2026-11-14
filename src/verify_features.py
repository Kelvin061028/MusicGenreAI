"""Check split CSVs, feature files, class counts and feature dimensions."""
import numpy as np
import pandas as pd

try:
    from .config import PROCESSED_DATA_DIR, GENRES
except ImportError:
    from config import PROCESSED_DATA_DIR, GENRES


def main():
    total = 0
    shapes = set()
    for split in ("train", "validation", "test"):
        csv_path = PROCESSED_DATA_DIR / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing {csv_path}. Run preprocessing first.")
        df = pd.read_csv(csv_path)
        missing_files = []
        for relative_path in df["feature_path"].astype(str):
            path = PROCESSED_DATA_DIR / relative_path
            if not path.exists():
                missing_files.append(str(path))
                continue
            arr = np.load(path)
            if arr.ndim != 2 or not np.isfinite(arr).all():
                raise ValueError(f"Invalid feature array: {path}; shape={arr.shape}")
            shapes.add(arr.shape)
        print(f"{split:10s}: {len(df):4d} tracks")
        print(df["genre"].value_counts().reindex(GENRES, fill_value=0).to_string())
        if missing_files:
            raise FileNotFoundError("Missing feature files:\n" + "\n".join(missing_files[:10]))
        total += len(df)

    print(f"\nTotal tracks across splits: {total}")
    print("Feature shapes:", sorted(shapes))
    if len(shapes) != 1:
        raise ValueError(f"Expected one fixed feature shape, found: {shapes}")
    print("\nFeature verification passed.")


if __name__ == "__main__":
    main()
