"""Prepare a balanced, reproducible FMA-small subset for this project.

FMA audio and its metadata are downloaded separately. By default this script
looks under data/fma_small and data/fma_metadata/tracks.csv; command-line paths
can point to the folders in which the user downloaded them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from .config import (
    DATA_DIR,
    FMAX,
    FMIN,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    SAMPLE_RATE,
)

SPLIT_ALIASES = {
    "training": "train",
    "train": "train",
    "validation": "validation",
    "valid": "validation",
    "test": "test",
}


def read_fma_metadata(metadata_path: Path) -> pd.DataFrame:
    tracks = pd.read_csv(metadata_path, index_col=0, header=[0, 1])
    required = [("track", "genre_top"), ("set", "subset"), ("set", "split")]
    missing = [column for column in required if column not in tracks.columns]
    if missing:
        raise ValueError(f"FMA metadata is missing required columns: {missing}")

    small_mask = tracks[("set", "subset")].astype(str).str.lower() == "small"
    small = tracks.loc[small_mask]
    result = pd.DataFrame({
        "track_id": small.index.astype(int),
        "genre": small[("track", "genre_top")].astype("string").str.strip().str.lower().to_numpy(),
        "split": small[("set", "split")].astype("string").str.strip().str.lower().map(SPLIT_ALIASES).to_numpy(),
    })
    return result.dropna(subset=["genre", "split"]).reset_index(drop=True)


def choose_balanced_subset(metadata: pd.DataFrame, max_per_genre: int) -> tuple[pd.DataFrame, list[str]]:
    genres = sorted(metadata["genre"].unique().tolist())
    if len(genres) != 8:
        raise ValueError(f"Expected FMA-small's 8 top-level genres, found {len(genres)}: {genres}")

    rng = np.random.default_rng(RANDOM_STATE)
    # Keep FMA's artist-disjoint official split proportions (80/10/10).
    target_per_split = {
        "train": int(max_per_genre * 0.8),
        "validation": int(max_per_genre * 0.1),
        "test": max_per_genre - int(max_per_genre * 0.8) - int(max_per_genre * 0.1),
    }
    selected = []
    for genre in genres:
        genre_rows = metadata[metadata["genre"] == genre]
        for split, requested in target_per_split.items():
            candidates = genre_rows[genre_rows["split"] == split]
            n = min(requested, len(candidates))
            if n:
                chosen_indices = rng.choice(candidates.index.to_numpy(), size=n, replace=False)
                selected.extend(candidates.loc[chosen_indices].to_dict("records"))

    return pd.DataFrame(selected).sort_values(["split", "genre", "track_id"]).reset_index(drop=True), genres


def audio_to_log_mel(path: Path) -> np.ndarray:
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=30.0)
    target_length = SAMPLE_RATE * 30
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    else:
        audio = audio[:target_length]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return ((log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", type=Path, default=DATA_DIR / "fma_small")
    parser.add_argument("--metadata", type=Path, default=DATA_DIR / "fma_metadata" / "tracks.csv")
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument(
        "--max-per-genre",
        type=int,
        default=300,
        help="Balanced cap per genre; 300 uses up to 2,400 tracks and keeps FMA's official splits.",
    )
    args = parser.parse_args()
    if args.max_per_genre < 10:
        raise SystemExit("--max-per-genre must be at least 10")
    if not args.metadata.exists():
        raise SystemExit(
            f"FMA tracks.csv was not found at {args.metadata}. Download and extract the "
            "FMA metadata bundle from https://github.com/mdeff/fma, then pass its tracks.csv path."
        )
    if not args.audio_dir.exists():
        raise SystemExit(f"FMA audio folder was not found: {args.audio_dir}")

    candidates = read_fma_metadata(args.metadata)
    selected, genres = choose_balanced_subset(candidates, args.max_per_genre)
    selected["label"] = selected["genre"].map({genre: i for i, genre in enumerate(genres)})
    feature_dir = args.output_dir / "log_mel"
    feature_dir.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []
    for index, item in enumerate(selected.to_dict("records"), start=1):
        track_id = int(item["track_id"])
        track_key = f"{track_id:06d}"
        audio_path = args.audio_dir / track_key[:3] / f"{track_key}.mp3"
        if not audio_path.exists():
            failures.append({"track_id": track_id, "error": "Audio file not found"})
            continue

        feature_name = f"fma_{track_id:06d}.npy"
        output_path = feature_dir / feature_name
        try:
            np.save(output_path, audio_to_log_mel(audio_path))
            rows.append({
                "track_id": f"fma_{track_id:06d}",
                "path": f"data/fma_small/{track_key[:3]}/{track_key}.mp3",
                "genre": item["genre"],
                "label": item["label"],
                "feature_path": f"log_mel/{feature_name}",
                "split": item["split"],
            })
        except Exception as error:
            failures.append({"track_id": track_id, "error": str(error)})

        if index % 50 == 0 or index == len(selected):
            print(f"Processed {index}/{len(selected)} selected FMA tracks")

    if not rows:
        raise SystemExit("No FMA tracks were processed. Check the audio directory and metadata paths.")

    frame = pd.DataFrame(rows)
    for split in ("train", "validation", "test"):
        frame[frame["split"] == split].to_csv(args.output_dir / f"{split}.csv", index=False)
    frame.to_csv(args.output_dir / "metadata.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(REPORTS_DIR / "fma_preprocessing_errors.csv", index=False)

    print(f"Genres (class order): {genres}")
    print(f"Successfully processed: {len(frame)} / {len(selected)}")
    print("Tracks per genre and split:")
    print(frame.groupby(["genre", "split"]).size().unstack(fill_value=0).to_string())
    print(f"Saved features and split CSVs under: {args.output_dir}")


if __name__ == "__main__":
    main()
