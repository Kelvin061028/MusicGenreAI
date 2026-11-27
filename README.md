# MusicGenre AI — Deep Learning Music Genre Classification

This project compares four deep-learning architectures for classifying music genres:

**The active experiment uses the GTZAN dataset.** The FMA-small preprocessing script is an optional alternative.

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

### Optional: switch to the FMA-small dataset

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

## Part B: development sequence

The active experiment uses GTZAN. The four phases below keep tuning on the validation split and reserve the held-out test split for the final comparison.

### Phase 1 — Model optimization

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

The completed GTZAN sweep selected these learning rates and track-level validation results:

| Model | Selected learning rate | Validation accuracy | Validation macro F1 |
|---|---:|---:|---:|
| CNN | 0.0005 | 84.56% | 0.8460 |
| CRNN | 0.0005 | 81.88% | 0.8207 |
| LSTM | 0.001 | 72.48% | 0.7195 |
| ViT | 0.0005 | 72.48% | 0.7309 |

Full trial records are in `reports/tuning/`. The final checkpoints are in `models/`.

Architecture tuning is available for the requested settings: CNN filter width/dropout/L2; CRNN filter width, both BiLSTM widths and recurrent dropout; ViT depth, attention heads, dropout and L2; and LSTM pooling, recurrent widths, dropout and L2. The full search runs four to five targeted configurations per model and ranks them by validation macro F1, with accuracy as the tie-breaker:

```powershell
python -m src.tune_architectures --epochs 20 --patience 5 --crops-per-track 6
```

This keeps the six-crop training exposure used for final training. For a faster screening pass, you can reduce the per-epoch crops and epoch cap:

```powershell
python -m src.tune_architectures --epochs 10 --patience 4 --crops-per-track 1
```

Trial results are written to `reports/tuning/architecture_tuning_summary.csv`; the selected validation configurations go to `reports/tuning/best_architectures.json`. Retrain all four final checkpoints with the selected settings, the normal six-crop protocol, and up to 80 epochs, then run the held-out test evaluation:

```powershell
python -m src.train_best_architectures
```

The short search is only a screening pass, not a substitute for the full search or final training.

Training samples are drawn as one random crop per track in each shuffled pass through the training split. The crop setting controls how many such passes contribute to an epoch, preserving the original training order while keeping the trial and final-training sample budgets explicit.

### GPU training on Windows with WSL2

The native Windows TensorFlow build in the project's `.venv` is CPU-only. The existing `.venv_wsl` was created with Python 3.14, which is outside the Python versions supported by the current TensorFlow wheel. Create a fresh WSL2 environment with a supported Python version (3.10–3.13) and the GPU-enabled TensorFlow extra:

```powershell
wsl
cd /mnt/c/Users/Kelvin/Downloads/MusicGenreAI_Week1_Starter/MusicGenreAI
python3.12 -m venv .venv_gpu
source .venv_gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
nvidia-smi
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
python -m src.tune_architectures --epochs 20 --patience 5 --crops-per-track 6
python -m src.train_best_architectures
```

If `python3.12` is not installed in WSL, install a supported Python version first or substitute an installed Python 3.10–3.13 executable. Confirm `nvidia-smi` works and the TensorFlow GPU list is non-empty before starting the sweep. Each training script prints the detected TensorFlow GPU devices and enables memory growth to fit within available VRAM.

### Phase 2 — Ensemble learning

The final evaluation script builds an equal-weight probability average from the CNN and CRNN outputs and compares it with each individual model. This is a fixed, transparent ensemble; the test set is not used to choose its weights.

### Phase 3 — System development

Run the Streamlit application:

```powershell
streamlit run app.py
```

Upload one track, choose any of the four classifiers, and view their genre probabilities and top-three results. When both CNN and CRNN are selected, the app also shows their ensemble. The app keeps up to 20 predictions in session history; clearing or ending the session removes that history.

### Phase 4 — Final evaluation

After final training is complete, evaluate the saved checkpoints against the held-out GTZAN test split:

```powershell
python -m src.evaluate_final
```

The script writes track-level accuracy, macro F1, weighted F1, per-class reports, confusion-matrix charts, and a model-comparison chart under `reports/`. It does not tune against the test split.

Latest final evaluation (151 held-out test tracks):

| Model | Test accuracy | Macro F1 |
|---|---:|---:|
| CNN | 83.44% | 0.8279 |
| CNN + CRNN ensemble | 82.78% | 0.8141 |
| CRNN | 80.79% | 0.7974 |
| ViT | 80.13% | 0.7972 |
| LSTM | 68.87% | 0.6703 |

The equal-weight ensemble did not outperform the CNN on this split. Keep it as a comparison option in the app and report this outcome as-is; do not claim that ensembling improved accuracy.

Generate the baseline model comparison charts and table with:

```powershell
python -m src.compare_models
```

This produces accuracy, macro F1, per-genre F1, and training-history comparisons under `reports/`.

## Current status

- [x] GTZAN train/validation/test split and feature preparation
- [x] Tune and train CNN, CRNN, LSTM and ViT
- [x] Compare CNN+CRNN ensemble against individual models
- [x] Streamlit upload, model selection, probability charts and session history
- [x] Final held-out test evaluation and comparison charts
