# MusicGenre AI — Deep Learning Music Genre Classification

This project compares four deep-learning architectures for classifying music genres:

1. CNN (Convolutional Neural Network)
2. CRNN (CNN + LSTM)
3. LSTM (Long Short-Term Memory)
4. Vision Transformer (ViT)

Traditional machine-learning classifiers such as KNN, Random Forest and SVM are intentionally excluded.

## Week 1 scope

- Set up the Python environment.
- Place the GTZAN audio files in the expected folder structure.
- Inspect the dataset.
- Convert each complete track into a standardised log-Mel spectrogram.
- Create stratified train/validation/test splits at the track level (70% / 15% / 15%).

## 1. Requirements

Recommended: Python 3.10 or 3.11. Use a virtual environment.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run the commands from Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

## 2. Dataset setup

Obtain the GTZAN Genre Collection from a legitimate research dataset source. Extract the audio files so the folder structure looks like this:

```text
data/
└── genres_original/
    ├── blues/
    │   ├── blues.00000.wav
    │   └── ...
    ├── classical/
    ├── country/
    ├── disco/
    ├── hiphop/
    ├── jazz/
    ├── metal/
    ├── pop/
    ├── reggae/
    └── rock/
```

The folder names must be lowercase and match the names above. Do not rename files into a different genre. The raw audio is not included in this starter project.

Dataset reference: https://github.com/mdeff/fma (FMA is an alternative dataset; the current Week 1 scripts are configured for GTZAN).

## 3. Inspect the dataset

From the project root:

```powershell
python -m src.inspect_dataset
```

Check that all ten genre folders are detected and that the file counts look reasonable.

## 4. Preprocess the audio

```powershell
python -m src.prepare_dataset
```

The script will:

- Load audio as mono at 22,050 Hz.
- Use the first 30 seconds of each track (padding shorter files).
- Generate a 128-bin log-Mel spectrogram.
- Standardise each spectrogram.
- Save one `.npy` feature array per track.
- Create stratified 70% / 15% / 15% CSV splits.

Output:

```text
data/processed/
├── log_mel/
├── metadata.csv
├── train.csv
├── validation.csv
└── test.csv
```

## Important experimental rule

Split by original track before creating any smaller windows or augmented versions. Never allow pieces derived from the same original track to appear in both training and test sets. This avoids data leakage and overly optimistic evaluation.

## Four-model experiment plan

| Model | Input | Main idea |
|---|---|---|
| CNN | Log-Mel spectrogram | Learns local time-frequency patterns |
| CRNN | Log-Mel spectrogram | CNN feature extraction + LSTM sequence learning |
| LSTM | MFCC or log-Mel time sequences | Learns temporal dependencies |
| ViT | Log-Mel spectrogram patches | Learns global relationships using self-attention |

All four models will use the same track-level split and evaluation metrics. Model-specific input transformations will be documented so comparisons remain fair.

## Current status

- [x] Project structure and configuration
- [x] Dataset inspection script
- [x] Log-Mel preprocessing script
- [ ] Run preprocessing on the downloaded dataset
- [ ] Implement and train CNN
- [ ] Implement and train CRNN
- [ ] Implement and train LSTM
- [ ] Implement and train ViT
- [ ] Compare models and build the application
