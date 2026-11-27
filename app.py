
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import librosa
import tempfile
import json
import hashlib
from datetime import datetime

from pathlib import Path

# =====================================================
# 1. CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
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
# 4. GENERATE LOG-MEL SPECTROGRAM
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


def extract_music_characteristics(audio):
    """Calculate descriptive audio features for the analyzed audio window."""
    # Read the tempo by position so this remains compatible with librosa
    # releases that return additional beat-tracking values.
    tempo_result = librosa.beat.beat_track(y=audio, sr=SAMPLE_RATE)
    tempo_value = tempo_result[0]
    tempo_array = np.asarray(tempo_value).reshape(-1)
    tempo = float(tempo_array[0]) if tempo_array.size else 0.0

    rms = float(np.mean(librosa.feature.rms(y=audio, frame_length=N_FFT, hop_length=HOP_LENGTH)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(
        y=audio, frame_length=N_FFT, hop_length=HOP_LENGTH)))
    chroma = librosa.feature.chroma_stft(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
    ).mean(axis=1)
    return {
        "tempo_bpm": tempo,
        "rms_energy": rms,
        "spectral_centroid_hz": centroid,
        "spectral_bandwidth_hz": bandwidth,
        "spectral_rolloff_hz": rolloff,
        "zero_crossing_rate": zcr,
        "chroma": np.asarray(chroma, dtype=float).tolist(),
    }


# =====================================================
# 5. AUTOMATIC MODEL INPUT PREPARATION
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


@st.cache_data(show_spinner=False)
def predict_uploaded_track(audio_bytes, audio_name, model_versions, norm_stats_json, genre_names, selected_models):
    """Run selected classifiers and the CNN+CRNN probability ensemble."""
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=Path(audio_name).suffix,
    ) as temp_file:
        temp_file.write(audio_bytes)
        temp_path = temp_file.name

    try:
        load_result = librosa.load(
            temp_path,
            sr=SAMPLE_RATE,
            mono=True,
            duration=DURATION,
        )
        audio = load_result[0]
    finally:
        Path(temp_path).unlink(missing_ok=True)

    target_length = SAMPLE_RATE * DURATION
    audio = audio[:target_length]
    feature_audio = audio if len(audio) else np.zeros(SAMPLE_RATE, dtype=np.float32)
    music_characteristics = extract_music_characteristics(feature_audio)
    if len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)))
    mel_db = generate_spectrogram(audio)
    norm_stats = json.loads(norm_stats_json)

    results = {}
    errors = {}
    for model_name, filename in MODEL_FILES.items():
        if model_name not in selected_models:
            continue
        model_path = MODEL_DIR / filename
        if not model_path.exists():
            errors[model_name] = f"Checkpoint not found: {filename}"
            continue
        try:
            model = load_model(str(model_path), model_versions[model_name])
            model_input = prepare_model_input(model, mel_db, norm_stats)
            segment_predictions = model.predict(model_input, verbose=0)
            probabilities = np.asarray(segment_predictions).mean(axis=0)
            if probabilities.shape[-1] != len(genre_names):
                raise ValueError(
                    f"{model_name} outputs {probabilities.shape[-1]} classes, "
                    f"but normalization metadata defines {len(genre_names)} genres."
                )
            results[model_name] = probabilities.astype(float)
        except Exception as error:
            errors[model_name] = str(error)

    if "CNN" in results and "CRNN" in results:
        results["CNN + CRNN Ensemble"] = ((results["CNN"] + results["CRNN"]) / 2).astype(float)
    return results, errors, music_characteristics


# =====================================================
from src.streamlit_pages import run_pages

run_pages({
    "st": st,
    "root": ROOT,
    "genres": GENRES,
    "duration": DURATION,
    "model_files": MODEL_FILES,
    "model_dir": MODEL_DIR,
    "norm_stats_path": NORM_STATS_PATH,
    "predict": predict_uploaded_track,
    "signature": lambda audio_bytes, selected_models: hashlib.sha256(
        audio_bytes + "|".join(selected_models).encode("utf-8")
    ).hexdigest(),
})
