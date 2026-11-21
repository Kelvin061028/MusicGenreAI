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

### Use the FMA-small dataset

FMA-small audio does not include its genre labels and official train/validation/test split in the audio ZIP. Download and extract the separate `fma_metadata.zip` from the [FMA dataset repository](https://github.com/mdeff/fma), then prepare a balanced subset from the official splits:

```powershell
python -m src.prepare_fma_dataset `
  --audio-dir "C:\Users\Kelvin\Downloads\fma_small" `
  --metadata "C:\Users\Kelvin\Downloads\fma_metadata\tracks.csv" `
  --max-per-genre 500
```

This selects up to 500 tracks per top-level genre (up to 4,000 total), retains FMA's official split assignments, and writes the generated features and split CSVs under `data/processed/`. The FMA-small task has eight genres, so retrain all four models after preparing it. Each artist's tracks remain in the same official split to reduce artist leakage. The audio files remain local; check each track's license before redistribution.

FMA dataset and metadata attribution: Defferrard et al., [FMA: A Dataset for Music Analysis](https://github.com/mdeff/fma). FMA metadata is distributed under CC BY 4.0; each audio track retains its artist-selected license.

For the original GTZAN preprocessing command above, the script will:

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

## Part B: validation-only hyperparameter tuning

Run a small learning-rate sweep for all four models from the project root:

```powershell
python -m src.tune_models --epochs 20 --patience 5
```

The shorter epoch cap and early-stopping patience apply only to the learning-rate sweep. Use the same settings for all four models during tuning. Final training still defaults to up to 80 epochs; choose the rates from the validation results, then train the final checkpoints.

The default sweep tries `1e-4`, `5e-4`, and `1e-3` for each architecture. It uses track-level validation accuracy and macro F1 to compare trials, and does not evaluate the test split during tuning. Checkpoints and histories are kept in separate `models/tuning/` and `reports/tuning/` subfolders so baseline models and reports are not overwritten.

To tune only selected models or use a different small candidate set:

```powershell
python -m src.tune_models --models cnn crnn --learning-rates 0.0001 0.0005 0.001
```

Review `reports/tuning/best_configurations.json` and `reports/tuning/validation_macro_f1_by_learning_rate.png`. Then retrain each selected model at its chosen learning rate without tuning mode, and run its normal evaluation once on the test split. Use the PowerShell environment variable `MUSICGENRE_LEARNING_RATE` to set that model's selected rate for the final run.

Generate the baseline model comparison charts and table with:

```powershell
python -m src.compare_models
```

This produces accuracy, macro F1, per-genre F1, and training-history comparisons under `reports/`.

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
