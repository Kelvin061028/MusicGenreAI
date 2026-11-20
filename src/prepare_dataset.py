"""Create log-Mel spectrogram arrays and track-level train/validation/test CSV splits."""
from pathlib import Path
import warnings

import librosa
import numpy as np
import pandas as pd

try:
    from .config import (
        GENRES, RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR,
        SAMPLE_RATE, DURATION_SECONDS, N_FFT, HOP_LENGTH, N_MELS,
        FMIN, FMAX, RANDOM_STATE,
    )
except ImportError:
    from config import (
        GENRES, RAW_DATA_DIR, PROCESSED_DATA_DIR, REPORTS_DIR,
        SAMPLE_RATE, DURATION_SECONDS, N_FFT, HOP_LENGTH, N_MELS,
        FMIN, FMAX, RANDOM_STATE,
    )

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


def find_audio_files() -> list[dict[str, str]]:
    rows = []
    for label, genre in enumerate(GENRES):
        genre_dir = RAW_DATA_DIR / genre
        if not genre_dir.exists():
            warnings.warn(f"Missing genre folder: {genre_dir}")
            continue
        for path in sorted(genre_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS:
                rows.append({"track_id": f"{genre}_{path.stem}", "path": str(path),
                             "genre": genre, "label": label})
    return rows


def audio_to_log_mel(path: str) -> np.ndarray:
    """Load one track and return a fixed-size [n_mels, time] log-Mel array."""
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=DURATION_SECONDS)
    target_length = SAMPLE_RATE * DURATION_SECONDS
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)))
    else:
        y = y[:target_length]

    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX, power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Per-track standardisation helps keep input scales consistent.
    mean = float(log_mel.mean())
    std = float(log_mel.std())
    log_mel = (log_mel - mean) / (std + 1e-6)
    return log_mel.astype(np.float32)


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = find_audio_files()
    if not rows:
        raise SystemExit(
            f"No audio files found in {RAW_DATA_DIR}. Run the dataset setup described in README.md."
        )

    metadata = []
    failures = []
    feature_dir = PROCESSED_DATA_DIR / "log_mel"
    feature_dir.mkdir(parents=True, exist_ok=True)

    for index, row in enumerate(rows, start=1):
        output_path = feature_dir / f"{row['track_id']}.npy"
        try:
            feature = audio_to_log_mel(row["path"])
            np.save(output_path, feature)
            # Store project-relative paths so the generated CSVs work across machines.
            metadata.append({
                **row,
                "path": Path(row["path"]).relative_to(RAW_DATA_DIR.parent.parent).as_posix(),
                "feature_path": output_path.relative_to(PROCESSED_DATA_DIR).as_posix(),
            })
        except Exception as exc:  # Keep processing other tracks if one file is corrupt.
            failures.append({"path": row["path"], "error": str(exc)})
        if index % 50 == 0 or index == len(rows):
            print(f"Processed {index}/{len(rows)} tracks")

    if not metadata:
        raise SystemExit("No tracks were processed successfully. Check audio files and dependencies.")

    df = pd.DataFrame(metadata)
    # Stratified split by complete track (70/15/15), without an ML library.
    # Splitting original tracks before any augmentation/windows prevents leakage.
    rng = np.random.default_rng(RANDOM_STATE)
    train_parts, val_parts, test_parts = [], [], []
    for _, genre_df in df.groupby("label", sort=True):
        indices = genre_df.index.to_numpy().copy()
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(n * 0.70)
        n_val = int(n * 0.15)
        train_parts.extend(indices[:n_train])
        val_parts.extend(indices[n_train:n_train + n_val])
        test_parts.extend(indices[n_train + n_val:])

    train_df = df.loc[train_parts].sample(frac=1, random_state=RANDOM_STATE)
    val_df = df.loc[val_parts].sample(frac=1, random_state=RANDOM_STATE)
    test_df = df.loc[test_parts].sample(frac=1, random_state=RANDOM_STATE)

    train_df.assign(split="train").to_csv(PROCESSED_DATA_DIR / "train.csv", index=False)
    val_df.assign(split="validation").to_csv(PROCESSED_DATA_DIR / "validation.csv", index=False)
    test_df.assign(split="test").to_csv(PROCESSED_DATA_DIR / "test.csv", index=False)
    df.to_csv(PROCESSED_DATA_DIR / "metadata.csv", index=False)

    if failures:
        pd.DataFrame(failures).to_csv(REPORTS_DIR / "preprocessing_errors.csv", index=False)

    print("\nPreprocessing complete")
    print(f"Successful tracks: {len(df)}")
    print(f"Failed tracks: {len(failures)}")
    print(f"Train / validation / test: {len(train_df)} / {len(val_df)} / {len(test_df)}")
    print("Class counts by split:")
    print(df.assign(split=["processed"] * len(df)).groupby("genre").size().to_string())
    print(f"\nSaved features and CSV files under: {PROCESSED_DATA_DIR}")


if __name__ == "__main__":
    main()
