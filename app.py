
import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import tempfile
import json

from pathlib import Path

# =====================================================
# 1. CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"

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
def load_model(model_path):

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
    ) / (std + 1e-8)

    return mel_db.astype(np.float32)


# =====================================================
# 6. AUTOMATIC MODEL INPUT PREPARATION
# =====================================================

def prepare_model_input(model, mel_db):

    input_shape = model.input_shape

    # Handle single-input models
    if isinstance(input_shape, list):
        if len(input_shape) != 1:
            raise ValueError(
                "This application supports single-input models only."
            )
        input_shape = input_shape[0]

    # -------------------------------------------------
    # Rank 4: Image input (CNN, CRNN, ViT)
    # Expected: (None, height, width, channels)
    # -------------------------------------------------

    if len(input_shape) == 4:

        expected_height = input_shape[1]
        expected_width = input_shape[2]
        expected_channels = input_shape[3]

        if expected_channels != 1:
            raise ValueError(
                f"Expected one input channel, got {expected_channels}"
            )

        image = mel_db[np.newaxis, ..., np.newaxis]

        # Resize to model's actual input dimensions
        if expected_height is not None and expected_width is not None:

            image = tf.image.resize(
                image,
                size=(
                    int(expected_height),
                    int(expected_width)
                )
            ).numpy()

        return image.astype(np.float32)

    # -------------------------------------------------
    # Rank 3: Sequential input (LSTM)
    # Expected: (None, timesteps, features)
    # -------------------------------------------------

    elif len(input_shape) == 3:

        expected_steps = input_shape[1]
        expected_features = input_shape[2]

        sequence = mel_db.T

        # Convert to expected feature orientation
        if expected_features == 128:
            pass

        elif expected_steps == 128:
            sequence = mel_db

        else:
            raise ValueError(
                f"Cannot determine sequence orientation: {input_shape}"
            )

        # Resize sequence length if required
        if expected_steps is not None:

            sequence = tf.image.resize(
                sequence[np.newaxis, ..., np.newaxis],
                size=(
                    int(expected_steps),
                    int(expected_features)
                )
            ).numpy()[0, ..., 0]

        return sequence[np.newaxis, ...].astype(
            np.float32
        )

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
                        str(model_path)
                    )

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
                        mel_db
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
                    predictions = model.predict(
                        model_input,
                        verbose=0
                    )[0]

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

st.caption(
    "MusicGenre AI | CNN | CRNN | LSTM | Vision Transformer"
)
