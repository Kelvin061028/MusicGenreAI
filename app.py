
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import librosa
import tempfile
import json

from pathlib import Path

# =====================================================
# 1. CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
PROCESSED_DIR = ROOT / "data" / "processed"
NORM_STATS_PATH = MODEL_DIR / "norm_stats.json"

GENRES = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
]

SAMPLE_RATE = 22050
DURATION = 30
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 20
FMAX = 8000

MODEL_FILES = {
    "CNN": "cnn_best.keras",
    "CRNN": "crnn_best.keras",
    "LSTM": "lstm_best.keras",
    "Vision Transformer": "vit_best.keras"
}

# =====================================================
# 2. CUSTOM LAYERS
# =====================================================


@tf.keras.utils.register_keras_serializable()
class SpecAugment(tf.keras.layers.Layer):

    def __init__(
        self,
        freq_mask=16,
        time_mask=16,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.freq_mask = freq_mask
        self.time_mask = time_mask

    def call(self, inputs, training=None):
        # Augmentation is disabled during inference.
        return inputs

    def get_config(self):

        config = super().get_config()

        config.update({
            "freq_mask": self.freq_mask,
            "time_mask": self.time_mask
        })

        return config



@tf.keras.utils.register_keras_serializable()
class PatchExtractor(tf.keras.layers.Layer):

    def call(self, images):

        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, 16, 16, 1],
            strides=[1, 16, 16, 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )

        batch_size = tf.shape(patches)[0]

        return tf.reshape(
            patches,
            [batch_size, -1, 256]
        )


@tf.keras.utils.register_keras_serializable()
class PatchEncoder(tf.keras.layers.Layer):

    def __init__(
        self,
        num_patches=128,
        projection_dim=128,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.num_patches = num_patches
        self.projection_dim = projection_dim

        self.projection = tf.keras.layers.Dense(
            projection_dim
        )

        self.position_embedding = (
            tf.keras.layers.Embedding(
                input_dim=num_patches,
                output_dim=projection_dim
            )
        )

    def call(self, patches):

        positions = tf.range(
            start=0,
            limit=self.num_patches,
            delta=1
        )

        return (
            self.projection(patches)
            + self.position_embedding(positions)
        )

    def get_config(self):

        config = super().get_config()

        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim
        })

        return config


CUSTOM_OBJECTS = {
    "SpecAugment": SpecAugment,
    "PatchExtractor": PatchExtractor,
    "PatchEncoder": PatchEncoder
}

# =====================================================
# 3. LOAD MODELS
# =====================================================

@st.cache_resource
def load_model(model_path, model_mtime_ns):

    model = tf.keras.models.load_model(
        model_path,
        custom_objects=CUSTOM_OBJECTS,
        compile=False
    )

    return model


# =====================================================
# 4. AUDIO PREPROCESSING
# =====================================================

def load_audio(uploaded_file):

    suffix = Path(uploaded_file.name).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(
            uploaded_file.getvalue()
        )

        temp_path = temp_file.name

    try:

        audio, sr = librosa.load(
            temp_path,
            sr=SAMPLE_RATE,
            mono=True,
            duration=DURATION
        )

    finally:

        Path(temp_path).unlink(
            missing_ok=True
        )

    target_length = SAMPLE_RATE * DURATION

    # Pad short audio
    if len(audio) < target_length:

        audio = np.pad(
            audio,
            (0, target_length - len(audio))
        )

    # Truncate long audio
    audio = audio[:target_length]

    return audio


# =====================================================
# 5. GENERATE LOG-MEL SPECTROGRAM
# =====================================================

def generate_spectrogram(audio):

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=FMIN,
        fmax=FMAX
    )

    mel_db = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Same per-track standardization
    mean = np.mean(mel_db)
    std = np.std(mel_db)

    mel_db = (
        mel_db - mean
    ) / (std + 1e-6)

    return mel_db.astype(np.float32)


# =====================================================
# 6. AUTOMATIC MODEL INPUT PREPARATION
# =====================================================

def prepare_model_input(model, mel_db, norm_stats):

    # Match training/evaluation: normalize with training-set statistics, then
    # split the full track into non-overlapping 3-second segments.
    segment_frames = int(norm_stats["segment_frames"])
    clip_frames = int(norm_stats["clip_frames"])
    mean = float(norm_stats["mean"])
    std = float(norm_stats["std"])

    if mel_db.shape[0] != int(norm_stats["n_mels"]):
        raise ValueError(
            f"Expected {norm_stats['n_mels']} Mel bins, got {mel_db.shape[0]}"
        )

    if mel_db.shape[1] < clip_frames:
        mel_db = np.pad(mel_db, ((0, 0), (0, clip_frames - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :clip_frames]

    mel_db = (mel_db - mean) / std
    num_segments = clip_frames // segment_frames
    usable_frames = num_segments * segment_frames
    segments = mel_db[:, :usable_frames].reshape(
        mel_db.shape[0], num_segments, segment_frames
    )
    segments = segments.transpose(1, 0, 2)[..., np.newaxis]

    input_shape = model.input_shape

    # Handle single-input models
    if isinstance(input_shape, list):
        if len(input_shape) != 1:
            raise ValueError(
                "This application supports single-input models only."
            )
        input_shape = input_shape[0]

    # -------------------------------------------------
    # Current CNN, CRNN, LSTM, and ViT models all accept segment images.
    # The LSTM converts each segment to a sequence inside its model graph.
    # -------------------------------------------------

    if len(input_shape) == 4:

        expected_height = input_shape[1]
        expected_width = input_shape[2]
        expected_channels = input_shape[3]

        if expected_channels != 1:
            raise ValueError(
                f"Expected one input channel, got {expected_channels}"
            )

        if (expected_height is not None and expected_height != segments.shape[1]) or (
            expected_width is not None and expected_width != segments.shape[2]
        ):
            raise ValueError(
                f"Model expects segment shape {input_shape[1:3]}, "
                f"but preprocessing produced {segments.shape[1:3]}"
            )

        return segments.astype(np.float32)

    # -------------------------------------------------
    # Rank 3: Sequential input (LSTM)
    # Expected: (None, timesteps, features)
    # -------------------------------------------------

    elif len(input_shape) == 3:

        expected_steps = input_shape[1]
        expected_features = input_shape[2]

        sequence = segments[..., 0].transpose(0, 2, 1)

        # Convert to expected feature orientation
        if expected_features == 128:
            pass

        elif expected_steps == 128:
            sequence = segments[..., 0]

        else:
            raise ValueError(
                f"Cannot determine sequence orientation: {input_shape}"
            )

        # Resize sequence length if required
        if expected_steps is not None:

            sequence = tf.image.resize(
                sequence[..., np.newaxis],
                size=(
                    int(expected_steps),
                    int(expected_features)
                )
            ).numpy()[..., 0]

        return sequence.astype(np.float32)

    else:

        raise ValueError(
            f"Unsupported model input shape: {input_shape}"
        )


# =====================================================
# 7. STREAMLIT INTERFACE
# =====================================================

st.set_page_config(
    page_title="MusicGenre AI",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 MusicGenre AI")

st.subheader(
    "Music Genre Classification Using Deep Learning"
)

st.write(
    "Upload a music track and classify it using "
    "four different deep learning architectures."
)

st.divider()

# Model selection
selected_model = st.selectbox(
    "Select Deep Learning Model",
    list(MODEL_FILES.keys())
)

# Audio upload
uploaded_file = st.file_uploader(
    "Upload a music file",
    type=["wav", "mp3", "ogg", "flac"]
)

# =====================================================
# 8. PREDICTION
# =====================================================

if uploaded_file is not None:

    st.audio(
        uploaded_file,
        format=uploaded_file.type
    )

    st.write(
        f"**Uploaded file:** {uploaded_file.name}"
    )

    model_path = (
        MODEL_DIR / MODEL_FILES[selected_model]
    )

    if not model_path.exists():

        st.warning(
            f"{selected_model} model has not been trained yet."
        )

        st.write(
            f"Expected file: `{model_path}`"
        )

    else:

        if st.button(
            "Predict Music Genre",
            type="primary"
        ):

            try:

                with st.spinner(
                    "Analyzing your music..."
                ):

                    # Load model
                    model = load_model(
                        str(model_path), model_path.stat().st_mtime_ns
                    )

                    if not NORM_STATS_PATH.exists():
                        raise FileNotFoundError(
                            f"Training normalization stats not found: {NORM_STATS_PATH}"
                        )
                    with NORM_STATS_PATH.open("r", encoding="utf-8") as stats_file:
                        norm_stats = json.load(stats_file)

                    # Process audio
                    audio = load_audio(
                        uploaded_file
                    )

                    mel_db = generate_spectrogram(
                        audio
                    )

                    # Prepare input automatically
                    model_input = prepare_model_input(
                        model,
                        mel_db,
                        norm_stats
                    )

                    # Debug information
                    st.write(
                        "**Model expected input:**",
                        model.input_shape
                    )

                    st.write(
                        "**Actual prediction input:**",
                        model_input.shape
                    )

                    # Predict
                    segment_predictions = model.predict(
                        model_input,
                        verbose=0
                    )
                    predictions = segment_predictions.mean(axis=0)

                    predicted_index = int(
                        np.argmax(predictions)
                    )

                    predicted_genre = GENRES[
                        predicted_index
                    ]

                    confidence = float(
                        predictions[predicted_index]
                    )

                # -------------------------------------
                # DISPLAY RESULTS
                # -------------------------------------

                st.success(
                    "Prediction completed successfully!"
                )

                st.divider()

                st.subheader(
                    "Prediction Result"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Predicted Genre",
                        predicted_genre.title()
                    )

                with col2:

                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%"
                    )

                st.divider()

                # -------------------------------------
                # PROBABILITY DISTRIBUTION
                # -------------------------------------

                st.subheader(
                    "Genre Probability Distribution"
                )

                chart_data = {
                    "Genre": [
                        genre.title()
                        for genre in GENRES
                    ],
                    "Probability": predictions
                }

                st.bar_chart(
                    chart_data,
                    x="Genre",
                    y="Probability"
                )

                # Ranked results
                st.subheader(
                    "Prediction Ranking"
                )

                ranking = np.argsort(
                    predictions
                )[::-1]

                for rank, index in enumerate(
                    ranking[:5],
                    start=1
                ):

                    st.write(
                        f"{rank}. {GENRES[index].title()} "
                        f"— {predictions[index] * 100:.2f}%"
                    )

                st.caption(
                    "Confidence is the model's predicted probability, "
                    "not a guarantee of correctness."
                )

            except Exception as error:

                st.error(
                    "Prediction failed."
                )

                st.exception(error)

st.divider()

# =====================================================
# 9. DATASET AND MODEL RESULTS
# =====================================================

st.header("Dataset and Model Results")
st.caption(
    "Explore the dataset splits, validation-only tuning trials, and the latest saved test reports."
)

split_rows = []
split_sizes = {}
for split_name, csv_name in (
    ("Train", "train.csv"),
    ("Validation", "validation.csv"),
    ("Test", "test.csv"),
):
    split_path = PROCESSED_DIR / csv_name
    if split_path.exists():
        split_frame = pd.read_csv(split_path)
        split_sizes[split_name] = len(split_frame)
        if "genre" in split_frame.columns:
            counts = split_frame["genre"].value_counts()
            split_rows.extend(
                {"Genre": genre, "Split": split_name, "Tracks": int(count)}
                for genre, count in counts.items()
            )

if split_sizes:
    count_columns = st.columns(3)
    for column, split_name in zip(count_columns, ("Train", "Validation", "Test")):
        column.metric(f"{split_name} tracks", f"{split_sizes.get(split_name, 0):,}")

if split_rows:
    split_chart = pd.DataFrame(split_rows).pivot(
        index="Genre", columns="Split", values="Tracks"
    ).reindex(index=GENRES, columns=["Train", "Validation", "Test"]).fillna(0)
    st.subheader("Tracks per Genre and Split")
    st.bar_chart(split_chart)
else:
    st.info("Dataset split CSVs were not found under data/processed/.")

st.subheader("Learning-Rate Tuning")
tuning_path = REPORT_DIR / "tuning" / "tuning_summary.csv"
if tuning_path.exists():
    tuning_frame = pd.read_csv(tuning_path)
    tuning_pivot = tuning_frame.pivot(
        index="learning_rate", columns="model", values="macro_f1"
    ).sort_index()
    st.caption("Track-level validation macro F1. The tuning sweep does not use the test split.")
    st.line_chart(tuning_pivot)

    best_rows = (
        tuning_frame.sort_values(
            ["model", "macro_f1", "track_validation_accuracy"],
            ascending=[True, False, False],
        )
        .groupby("model", as_index=False)
        .first()
    )
    st.write("Best validation setting for each model")
    st.dataframe(
        best_rows[["model", "learning_rate", "track_validation_accuracy", "macro_f1"]]
        .rename(columns={
            "model": "Model",
            "learning_rate": "Learning rate",
            "track_validation_accuracy": "Validation accuracy",
            "macro_f1": "Validation macro F1",
        }),
        hide_index=True,
        use_container_width=True,
    )
else:
    st.info("No tuning summary found. Run `python -m src.tune_models` to create it.")

st.subheader("Latest Saved Test Results")
st.caption(
    "Loaded from reports/*_test_metrics.json. After final training with the selected rates, "
    "rerun model evaluation to refresh these test results."
)

test_rows = []
class_f1_rows = []
available_confusion_matrices = {}
for model_key, display_name, stem in (
    ("CNN", "CNN", "cnn"),
    ("CRNN", "CRNN", "crnn"),
    ("LSTM", "LSTM", "lstm"),
    ("Vision Transformer", "ViT", "vit"),
):
    metrics_path = REPORT_DIR / f"{stem}_test_metrics.json"
    if not metrics_path.exists():
        continue
    with metrics_path.open("r", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)
    report = metrics.get("classification_report", {})
    macro = report.get("macro avg", {})
    test_rows.append({
        "Model": display_name,
        "Accuracy": metrics.get("test_accuracy"),
        "Macro F1": macro.get("f1-score"),
    })
    for genre in metrics.get("class_names", []):
        class_f1_rows.append({
            "Genre": genre,
            "Model": display_name,
            "F1": report.get(genre, {}).get("f1-score"),
        })
    confusion_path = REPORT_DIR / f"{stem}_confusion_matrix.png"
    if confusion_path.exists():
        available_confusion_matrices[display_name] = confusion_path

if test_rows:
    test_frame = pd.DataFrame(test_rows).set_index("Model")
    st.bar_chart(test_frame)
else:
    st.info("No saved test metric reports were found under reports/.")

if class_f1_rows:
    class_f1_pivot = pd.DataFrame(class_f1_rows).pivot(
        index="Genre", columns="Model", values="F1"
    ).reindex(GENRES)
    st.subheader("Test F1 by Genre")
    st.bar_chart(class_f1_pivot)

if available_confusion_matrices:
    confusion_model = st.selectbox(
        "Confusion matrix",
        list(available_confusion_matrices),
        key="results_confusion_model",
    )
    st.image(
        str(available_confusion_matrices[confusion_model]),
        caption=f"{confusion_model} saved test confusion matrix",
        use_container_width=True,
    )

st.caption(
    "The prediction selector loads models/*.keras. Tuning checkpoints in models/tuning/ "
    "are separate; retrain the selected settings to update the app's models."
)

st.divider()

st.caption(
    "MusicGenre AI | CNN | CRNN | LSTM | Vision Transformer"
)
