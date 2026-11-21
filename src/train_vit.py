from pathlib import Path
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
FEATURE_DIR = DATA_DIR / "log_mel"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)
RUN_DIR = os.environ.get("MUSICGENRE_RUN_DIR")
ARTIFACT_MODEL_DIR = MODEL_DIR / RUN_DIR if RUN_DIR else MODEL_DIR
ARTIFACT_REPORT_DIR = REPORT_DIR / RUN_DIR if RUN_DIR else REPORT_DIR
ARTIFACT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- COMMON SETTINGS
SEED = 42
BATCH_SIZE = 32
EPOCHS = int(os.environ.get("MUSICGENRE_EPOCHS", "80"))
EARLY_STOPPING_PATIENCE = int(os.environ.get("MUSICGENRE_EARLY_STOPPING_PATIENCE", "14"))
LR_PATIENCE = int(os.environ.get("MUSICGENRE_LR_PATIENCE", "5"))
L2 = 1e-4
CLIP_SECONDS = 30
SEGMENT_SECONDS = 3
CROPS_PER_TRACK = 6
# ---------------------------------------------------------- MODEL-SPECIFIC SETTINGS
LEARNING_RATE = float(os.environ.get("MUSICGENRE_LEARNING_RATE", "5e-4"))  # FMA validation choice
IMAGE_HEIGHT = 128     # spectrogram is resized to (128 x 128) before patching
IMAGE_WIDTH = 128      # (3s crop is ~129 frames wide, so this is almost no distortion)
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_HEIGHT // PATCH_SIZE) * (IMAGE_WIDTH // PATCH_SIZE)  # 8 x 8 = 64
PROJECTION_DIM = 128
NUM_HEADS = 4
TRANSFORMER_LAYERS = 4
MLP_DIM = 256

tf.keras.utils.set_random_seed(SEED)
rng = np.random.default_rng(SEED)


# ----------------------------------------------------------------- data loading
def find_column(df, candidates, description):
    lower = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]
    raise ValueError(f"Cannot identify {description} column. Available: {list(df.columns)}")


def resolve_feature_path(value):
    p = Path(str(value).replace("\\", "/"))
    if p.suffix.lower() != ".npy":
        p = p.with_suffix(".npy")
    if p.is_absolute() and p.exists():
        return p
    candidate = FEATURE_DIR / p.name
    if candidate.exists():
        return candidate
    matches = list(FEATURE_DIR.rglob(p.name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Feature file not found for CSV value: {value}")


def load_split(csv_name):
    csv_path = DATA_DIR / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}")
    df = pd.read_csv(csv_path)
    path_col = find_column(
        df,
        ["feature_path", "features_path", "npy_path", "feature_file", "spectrogram_path",
         "processed_path", "filepath", "file_path", "path", "filename", "file"],
        "feature path",
    )
    label_col = find_column(df, ["genre", "label", "class", "target"], "genre/label")
    paths = [resolve_feature_path(v) for v in df[path_col]]
    labels = df[label_col].astype(str).str.strip().str.lower().to_numpy()
    return paths, labels


train_paths, train_txt = load_split("train.csv")
val_paths, val_txt = load_split("validation.csv")
test_paths, test_txt = load_split("test.csv")

class_names = sorted(set(train_txt) | set(val_txt) | set(test_txt))
class_to_id = {n: i for i, n in enumerate(class_names)}
num_classes = len(class_names)

def encode(labels):
    return np.array([class_to_id[x] for x in labels], dtype=np.int32)


y_train, y_val, y_test = map(encode, [train_txt, val_txt, test_txt])

first = np.load(train_paths[0])
if first.ndim != 2:
    raise ValueError(f"Expected 2D log-mel features, got {first.shape}")
N_MELS, N_FRAMES = first.shape  # features are stored as (mels, frames)
SEG = max(8, int(N_FRAMES * SEGMENT_SECONDS / CLIP_SECONDS))
N_SEG = N_FRAMES // SEG
INPUT_SHAPE = (N_MELS, SEG, 1)


def load_all(paths):
    arr = []
    for p in paths:
        x = np.load(p).astype(np.float32)
        if x.shape != (N_MELS, N_FRAMES):
            raise ValueError(f"Inconsistent shape in {p}: {x.shape}, expected {(N_MELS, N_FRAMES)}")
        arr.append(x)
    return np.stack(arr)[..., np.newaxis]


print("Loading features into memory ...")
X_train, X_val, X_test = load_all(train_paths), load_all(val_paths), load_all(test_paths)

# Standardize with TRAIN statistics only (identical to the other scripts).
MEAN, STD = float(X_train.mean()), float(X_train.std() + 1e-8)
for arr in (X_train, X_val, X_test):
    arr -= MEAN
    arr /= STD
with open(MODEL_DIR / "norm_stats.json", "w", encoding="utf-8") as f:
    json.dump({"mean": MEAN, "std": STD, "n_mels": N_MELS, "segment_frames": SEG,
               "clip_frames": N_FRAMES, "class_names": class_names}, f, indent=2)

print(f"Full feature: ({N_MELS}, {N_FRAMES}) -> segment input {INPUT_SHAPE}, {N_SEG} segments/track")
print("Classes:", class_names)
print(f"Tracks: train={len(X_train)}, validation={len(X_val)}, test={len(X_test)}")


def make_segments(X):
    """Non-overlapping segments, ordered track by track: (N*N_SEG, mels, SEG, 1)."""
    n = len(X)
    s = X[:, :, : N_SEG * SEG, 0].reshape(n, N_MELS, N_SEG, SEG)
    s = s.transpose(0, 2, 1, 3)[..., np.newaxis]
    return np.ascontiguousarray(s.reshape(n * N_SEG, N_MELS, SEG, 1))


Xv_seg, yv_seg = make_segments(X_val), np.repeat(y_val, N_SEG)
Xt_seg, yt_seg = make_segments(X_test), np.repeat(y_test, N_SEG)


def train_generator():
    """Endless random 3s crops from full training clips."""
    n = len(X_train)
    while True:
        for i in rng.permutation(n):
            start = rng.integers(0, N_FRAMES - SEG + 1)
            yield X_train[i, :, start:start + SEG, :], y_train[i]


train_ds = (
    tf.data.Dataset.from_generator(
        train_generator,
        output_signature=(tf.TensorSpec(INPUT_SHAPE, tf.float32), tf.TensorSpec((), tf.int32)),
    )
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)
STEPS_PER_EPOCH = (len(X_train) * CROPS_PER_TRACK) // BATCH_SIZE
val_ds = tf.data.Dataset.from_tensor_slices((Xv_seg, yv_seg)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ----------------------------------------------------------------------- model
@tf.keras.utils.register_keras_serializable()
class SpecAugment(tf.keras.layers.Layer):
    """Zero one frequency band and one time block per sample (training only)."""

    def __init__(self, freq_mask=16, time_mask=16, **kwargs):
        super().__init__(**kwargs)
        self.freq_mask = freq_mask
        self.time_mask = time_mask

    def call(self, inputs, training=None):
        if not training:
            return inputs
        x = inputs
        b, h, w = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2]
        f = tf.random.uniform([b], 0, self.freq_mask + 1, dtype=tf.int32)
        f0 = tf.random.uniform([b], 0, tf.maximum(1, h - self.freq_mask), dtype=tf.int32)
        t = tf.random.uniform([b], 0, self.time_mask + 1, dtype=tf.int32)
        t0 = tf.random.uniform([b], 0, tf.maximum(1, w - self.time_mask), dtype=tf.int32)
        rows = tf.range(h)[None, :, None, None]
        cols = tf.range(w)[None, None, :, None]
        fmask = (rows >= f0[:, None, None, None]) & (rows < (f0 + f)[:, None, None, None])
        tmask = (cols >= t0[:, None, None, None]) & (cols < (t0 + t)[:, None, None, None])
        return tf.where(fmask | tmask, tf.zeros_like(x), x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"freq_mask": self.freq_mask, "time_mask": self.time_mask})
        return cfg


@tf.keras.utils.register_keras_serializable()
class PatchExtractor(tf.keras.layers.Layer):
    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
            strides=[1, PATCH_SIZE, PATCH_SIZE, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        return tf.reshape(patches, [tf.shape(patches)[0], NUM_PATCHES, PATCH_SIZE * PATCH_SIZE])


@tf.keras.utils.register_keras_serializable()
class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = tf.keras.layers.Dense(
            projection_dim, kernel_regularizer=tf.keras.regularizers.l2(L2))
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim)

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patches) + self.position_embedding(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_patches": self.num_patches, "projection_dim": self.projection_dim})
        return cfg


def transformer_block(inputs, num_heads, projection_dim, mlp_dim):
    reg = tf.keras.regularizers.l2(L2)
    h = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    h = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=projection_dim // num_heads,
        dropout=0.1, kernel_regularizer=reg)(h, h)
    h = tf.keras.layers.Dropout(0.1)(h)
    x = tf.keras.layers.Add()([inputs, h])

    m = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    m = tf.keras.layers.Dense(mlp_dim, activation="gelu", kernel_regularizer=reg)(m)
    m = tf.keras.layers.Dropout(0.1)(m)
    m = tf.keras.layers.Dense(projection_dim, kernel_regularizer=reg)(m)
    m = tf.keras.layers.Dropout(0.1)(m)
    return tf.keras.layers.Add()([x, m])


def build_vit(input_shape, classes):
    inputs = tf.keras.Input(shape=input_shape, name="log_mel_spectrogram")
    x = SpecAugment(freq_mask=max(4, N_MELS // 8), time_mask=max(4, SEG // 8), name="spec_augment")(inputs)

    # Resize the 3s crop to (IMAGE_HEIGHT x IMAGE_WIDTH), then split into patches
    x = tf.keras.layers.Resizing(IMAGE_HEIGHT, IMAGE_WIDTH)(x)
    x = PatchExtractor()(x)
    x = PatchEncoder(NUM_PATCHES, PROJECTION_DIM)(x)

    for _ in range(TRANSFORMER_LAYERS):
        x = transformer_block(x, NUM_HEADS, PROJECTION_DIM, MLP_DIM)

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Same classifier head as the CNN / CRNN / LSTM
    x = tf.keras.layers.Dense(128, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(L2))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(classes, activation="softmax", name="genre_probabilities")(x)

    model = tf.keras.Model(inputs, outputs, name="MusicGenre_ViT_v2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


model = build_vit(INPUT_SHAPE, num_classes)
model.summary()

best_model_path = ARTIFACT_MODEL_DIR / "vit_best.keras"
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        best_model_path, monitor="val_loss", save_best_only=True, mode="min", verbose=1),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=LR_PATIENCE, min_lr=1e-6, verbose=1),
    tf.keras.callbacks.CSVLogger(ARTIFACT_REPORT_DIR / "vit_training_history.csv"),
]

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    callbacks=callbacks,
    verbose=1,
)

# ------------------------------------------------------------------ evaluation
best_model = tf.keras.models.load_model(
    best_model_path,
    custom_objects={"SpecAugment": SpecAugment, "PatchExtractor": PatchExtractor,
                    "PatchEncoder": PatchEncoder},
)

if os.environ.get("MUSICGENRE_TUNING_MODE") == "1":
    val_seg_probs = best_model.predict(Xv_seg, batch_size=BATCH_SIZE, verbose=1)
    val_track_probs = val_seg_probs.reshape(len(X_val), N_SEG, num_classes).mean(axis=1)
    val_predictions = np.argmax(val_track_probs, axis=1)
    val_report = classification_report(
        y_val, val_predictions, labels=np.arange(num_classes),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    with open(ARTIFACT_REPORT_DIR / "validation_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "learning_rate": LEARNING_RATE,
            "track_validation_accuracy": float(np.mean(val_predictions == y_val)),
            "macro_f1": float(val_report["macro avg"]["f1-score"]),
            "classification_report": val_report,
        }, f, indent=2)
    print(f"Validation track accuracy: {np.mean(val_predictions == y_val):.4f}")
    print(f"Validation macro F1: {val_report['macro avg']['f1-score']:.4f}")
    raise SystemExit(0)

seg_loss, seg_acc = best_model.evaluate(Xt_seg, yt_seg, batch_size=BATCH_SIZE, verbose=1)
seg_probs = best_model.predict(Xt_seg, batch_size=BATCH_SIZE, verbose=1)

# Average segment probabilities per track (segments are ordered track by track).
track_probs = seg_probs.reshape(len(X_test), N_SEG, num_classes).mean(axis=1)
predictions = np.argmax(track_probs, axis=1)
track_acc = float(np.mean(predictions == y_test))

report = classification_report(
    y_test, predictions, labels=np.arange(num_classes),
    target_names=class_names, output_dict=True, zero_division=0)
with open(ARTIFACT_REPORT_DIR / "vit_test_metrics.json", "w", encoding="utf-8") as f:
    json.dump({
        "segment_test_loss": float(seg_loss),
        "segment_test_accuracy": float(seg_acc),
        "track_test_accuracy": track_acc,
        "test_accuracy": track_acc,
        "class_names": class_names,
        "classification_report": report,
    }, f, indent=2)

cm = confusion_matrix(y_test, predictions, labels=np.arange(num_classes))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
ax.set_title("ViT — GTZAN Test Confusion Matrix (track level)")
fig.tight_layout()
fig.savefig(ARTIFACT_REPORT_DIR / "vit_confusion_matrix.png", dpi=160)
plt.close(fig)

print("\nViT training complete.")
print(f"Segment-level test accuracy: {seg_acc:.4f}")
print(f"Track-level test accuracy:   {track_acc:.4f}")
print(f"Best model: {best_model_path}")
