from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "genres_original"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock",
]

SAMPLE_RATE = 22_050
DURATION_SECONDS = 30
N_FFT = 2_048
HOP_LENGTH = 512
N_MELS = 128
FMIN = 20
FMAX = 8_000
RANDOM_STATE = 42

# A 30-second track at this sample rate and hop length produces about 1,292 frames.
EXPECTED_FRAMES = 1 + int(SAMPLE_RATE * DURATION_SECONDS / HOP_LENGTH)
