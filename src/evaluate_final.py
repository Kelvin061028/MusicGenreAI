"""Evaluate the four saved GTZAN models and CNN+CRNN ensemble on the test split."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
FEATURE_DIR = DATA_DIR / "log_mel"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)
BATCH_SIZE = 32
MODEL_FILES = {
    "CNN": "cnn_best.keras", "CRNN": "crnn_best.keras",
    "LSTM": "lstm_best.keras", "Vision Transformer": "vit_best.keras",
}


@tf.keras.utils.register_keras_serializable()
class SpecAugment(tf.keras.layers.Layer):
    def __init__(self, freq_mask=16, time_mask=16, **kwargs):
        super().__init__(**kwargs)
        self.freq_mask, self.time_mask = freq_mask, time_mask

    def call(self, inputs, training=None):
        return inputs

    def get_config(self):
        return {**super().get_config(), "freq_mask": self.freq_mask,
                "time_mask": self.time_mask}


@tf.keras.utils.register_keras_serializable()
class PatchExtractor(tf.keras.layers.Layer):
    def call(self, images):
        patches = tf.image.extract_patches(images, [1, 16, 16, 1],
                                           [1, 16, 16, 1], [1, 1, 1, 1], "VALID")
        return tf.reshape(patches, [tf.shape(patches)[0], -1, 256])


@tf.keras.utils.register_keras_serializable()
class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches=128, projection_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.num_patches, self.projection_dim = num_patches, projection_dim
        self.projection = tf.keras.layers.Dense(projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(num_patches, projection_dim)

    def call(self, patches):
        positions = tf.range(self.num_patches)
        return self.projection(patches) + self.position_embedding(positions)

    def get_config(self):
        return {**super().get_config(), "num_patches": self.num_patches,
                "projection_dim": self.projection_dim}


def find_column(frame, candidates):
    columns = {str(column).lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    raise ValueError(f"Required column not found. Available columns: {list(frame.columns)}")


def main():
    stats = json.loads((MODEL_DIR / "norm_stats.json").read_text(encoding="utf-8"))
    class_names = stats["class_names"]
    frame = pd.read_csv(DATA_DIR / "test.csv")
    path_col = find_column(frame, ["feature_path", "npy_path", "spectrogram_path",
                                  "processed_path", "filepath", "file_path", "path",
                                  "filename", "file"])
    label_col = find_column(frame, ["genre", "label", "class", "target"])
    labels = frame[label_col].astype(str).str.strip().str.lower().to_numpy()
    class_to_id = {name: index for index, name in enumerate(class_names)}
    y_true = np.asarray([class_to_id[label] for label in labels], dtype=np.int32)
    arrays = []
    for value in frame[path_col]:
        item = Path(str(value).replace("\\", "/"))
        path = item if item.is_absolute() and item.exists() else FEATURE_DIR / item.name
        if not path.exists():
            raise FileNotFoundError(f"Missing test feature: {value} (looked for {path})")
        arrays.append(np.load(path).astype(np.float32))
    features = np.stack(arrays)
    if features.shape[1:] != (int(stats["n_mels"]), int(stats["clip_frames"])):
        raise ValueError(f"Unexpected feature shape {features.shape[1:]}; check GTZAN stats.")
    features = (features - float(stats["mean"])) / float(stats["std"])
    segment_frames = int(stats["segment_frames"])
    segment_count = int(stats["clip_frames"]) // segment_frames
    features = features[:, :, :segment_count * segment_frames]
    segments = features.reshape(len(features), features.shape[1], segment_count,
                                segment_frames).transpose(0, 2, 1, 3)
    segments = segments.reshape(-1, features.shape[1], segment_frames, 1)

    probabilities = {}
    custom_objects = {"SpecAugment": SpecAugment, "PatchExtractor": PatchExtractor,
                      "PatchEncoder": PatchEncoder}
    for name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint missing: {path}")
        print(f"Evaluating {name} ...")
        model = tf.keras.models.load_model(path, custom_objects=custom_objects, compile=False)
        segment_probs = model.predict(segments, batch_size=BATCH_SIZE, verbose=1)
        track_probs = segment_probs.reshape(len(features), segment_count, -1).mean(axis=1)
        probabilities[name] = track_probs

    probabilities["CNN + CRNN Ensemble"] = (probabilities["CNN"] + probabilities["CRNN"]) / 2
    summary = []
    for name, probs in probabilities.items():
        predictions = probs.argmax(axis=1)
        report = classification_report(y_true, predictions, labels=np.arange(len(class_names)),
                                       target_names=class_names, output_dict=True,
                                       zero_division=0)
        summary.append({"Model": name, "Test accuracy": float(accuracy_score(y_true, predictions)),
                        "Macro F1": float(report["macro avg"]["f1-score"]),
                        "Weighted F1": float(report["weighted avg"]["f1-score"])})
        slug = "ensemble" if name == "CNN + CRNN Ensemble" else name.lower().replace(" ", "_")
        payload = {"model": name, "evaluation_split": "GTZAN held-out test",
                   "test_tracks": len(y_true), "test_accuracy": float(accuracy_score(y_true, predictions)),
                   "class_names": class_names, "classification_report": report}
        (REPORT_DIR / f"{slug}_test_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        matrix = confusion_matrix(y_true, predictions, labels=np.arange(len(class_names)))
        fig, ax = plt.subplots(figsize=(9, 7))
        ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(
            ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
        ax.set_title(f"{name} — GTZAN held-out test (track level)")
        fig.tight_layout()
        fig.savefig(REPORT_DIR / f"{slug}_final_test_confusion_matrix.png", dpi=160)
        plt.close(fig)

    summary_frame = pd.DataFrame(summary).sort_values("Test accuracy", ascending=False)
    summary_frame.to_csv(REPORT_DIR / "final_test_comparison.csv", index=False)
    ax = summary_frame.set_index("Model")[["Test accuracy", "Macro F1", "Weighted F1"]].plot(
        kind="bar", figsize=(10, 6), ylim=(0, 1), rot=20)
    ax.set_ylabel("Score")
    ax.set_title("GTZAN held-out test comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "final_test_comparison.png", dpi=160)
    plt.close()
    print("\nFinal track-level test results (test set was not used for tuning):")
    print(summary_frame.to_string(index=False, formatters={
        "Test accuracy": "{:.3f}".format, "Macro F1": "{:.3f}".format,
        "Weighted F1": "{:.3f}".format}))
    print(f"Reports saved in {REPORT_DIR}")


if __name__ == "__main__":
    main()
