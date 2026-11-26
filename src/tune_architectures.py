"""Validation-only architecture and regularization search for all four models.

Each model gets a compact, documented candidate design. Candidate ranking uses
track-level validation macro F1 (accuracy breaks ties); the test split is never
evaluated here. Selected settings are written to reports/tuning/.

Example: python -m src.tune_architectures --epochs 20 --patience 5 --crops-per-track 6
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "tuning"

# Compact one-factor-at-a-time style designs around the existing LR-selected
# checkpoints. Each architecture-specific parameter shown in the Phase 1 plan
# varies in at least one candidate while the rest stay near the baseline.
CANDIDATES: dict[str, list[dict[str, object]]] = {
    "cnn": [
        {"learning_rate": 5e-4, "filter_base": 32, "dropout": .25, "head_dropout": .5, "l2": 1e-4},
        {"learning_rate": 5e-4, "filter_base": 48, "dropout": .25, "head_dropout": .5, "l2": 1e-4},
        {"learning_rate": 5e-4, "filter_base": 32, "dropout": .4, "head_dropout": .5, "l2": 1e-3},
        {"learning_rate": 1e-3, "filter_base": 32, "dropout": .25, "head_dropout": .5, "l2": 1e-4},
    ],
    "crnn": [
        {"learning_rate": 5e-4, "filter_base": 32, "dropout": .25, "head_dropout": .5, "rnn_units_1": 96, "rnn_units_2": 64, "rnn_dropout": .2, "l2": 1e-4},
        {"learning_rate": 5e-4, "filter_base": 24, "dropout": .25, "head_dropout": .5, "rnn_units_1": 96, "rnn_units_2": 64, "rnn_dropout": .2, "l2": 1e-4},
        {"learning_rate": 5e-4, "filter_base": 32, "dropout": .25, "head_dropout": .5, "rnn_units_1": 64, "rnn_units_2": 48, "rnn_dropout": .2, "l2": 1e-4},
        {"learning_rate": 5e-4, "filter_base": 32, "dropout": .35, "head_dropout": .55, "rnn_units_1": 96, "rnn_units_2": 64, "rnn_dropout": .3, "l2": 1e-3},
        {"learning_rate": 1e-3, "filter_base": 32, "dropout": .25, "head_dropout": .5, "rnn_units_1": 96, "rnn_units_2": 64, "rnn_dropout": .2, "l2": 1e-4},
    ],
    "vit": [
        {"learning_rate": 5e-4, "transformer_layers": 4, "num_heads": 4, "mlp_dim": 256, "dropout": .1, "head_dropout": .5, "l2": 1e-4},
        {"learning_rate": 5e-4, "transformer_layers": 2, "num_heads": 4, "mlp_dim": 256, "dropout": .1, "head_dropout": .5, "l2": 1e-4},
        {"learning_rate": 5e-4, "transformer_layers": 4, "num_heads": 2, "mlp_dim": 256, "dropout": .1, "head_dropout": .5, "l2": 1e-4},
        {"learning_rate": 5e-4, "transformer_layers": 4, "num_heads": 4, "mlp_dim": 256, "dropout": .2, "head_dropout": .5, "l2": 1e-3},
        {"learning_rate": 1e-3, "transformer_layers": 4, "num_heads": 4, "mlp_dim": 256, "dropout": .1, "head_dropout": .5, "l2": 1e-4},
    ],
    "lstm": [
        {"learning_rate": 1e-3, "pool_size": 4, "lstm_units_1": 64, "lstm_units_2": 64, "dropout": .3, "head_dropout": .5, "l2": 1e-4},
        {"learning_rate": 1e-3, "pool_size": 2, "lstm_units_1": 64, "lstm_units_2": 64, "dropout": .3, "head_dropout": .5, "l2": 1e-4},
        {"learning_rate": 1e-3, "pool_size": 4, "lstm_units_1": 96, "lstm_units_2": 64, "dropout": .3, "head_dropout": .5, "l2": 1e-4},
        {"learning_rate": 1e-3, "pool_size": 4, "lstm_units_1": 64, "lstm_units_2": 32, "dropout": .4, "head_dropout": .5, "l2": 1e-3},
        {"learning_rate": 5e-4, "pool_size": 4, "lstm_units_1": 64, "lstm_units_2": 64, "dropout": .3, "head_dropout": .5, "l2": 1e-4},
    ],
}

ENV_KEYS = {
    "learning_rate": "MUSICGENRE_LEARNING_RATE", "filter_base": "MUSICGENRE_FILTER_BASE",
    "dropout": "MUSICGENRE_DROPOUT", "head_dropout": "MUSICGENRE_HEAD_DROPOUT",
    "l2": "MUSICGENRE_L2", "rnn_units_1": "MUSICGENRE_RNN_UNITS_1",
    "rnn_units_2": "MUSICGENRE_RNN_UNITS_2", "rnn_dropout": "MUSICGENRE_RNN_DROPOUT",
    "pool_size": "MUSICGENRE_POOL_SIZE", "lstm_units_1": "MUSICGENRE_LSTM_UNITS_1",
    "lstm_units_2": "MUSICGENRE_LSTM_UNITS_2",
    "transformer_layers": "MUSICGENRE_TRANSFORMER_LAYERS",
    "num_heads": "MUSICGENRE_NUM_HEADS", "mlp_dim": "MUSICGENRE_MLP_DIM",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=tuple(CANDIDATES), default=list(CANDIDATES))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--crops-per-track", type=int, choices=(1, 2, 3, 6), default=6,
                        help="Random training crops per track per epoch; defaults to final-training exposure.")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        parser.error("--epochs and --patience must be positive integers")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name in args.models:
        for trial, config in enumerate(CANDIDATES[model_name], start=1):
            run_dir = Path("tuning") / "architecture" / model_name / f"trial_{trial:02d}"
            metrics_path = ROOT / "reports" / run_dir / "validation_metrics.json"
            env = os.environ.copy()
            for key, value in config.items():
                env[ENV_KEYS[key]] = str(value)
            env["MUSICGENRE_EPOCHS"] = str(args.epochs)
            env["MUSICGENRE_EARLY_STOPPING_PATIENCE"] = str(args.patience)
            env["MUSICGENRE_LR_PATIENCE"] = str(max(2, args.patience // 2))
            # Keep all trials on the same sampling budget. Final training defaults to six.
            env["MUSICGENRE_CROPS_PER_TRACK"] = str(args.crops_per_track)
            env["MUSICGENRE_RUN_DIR"] = run_dir.as_posix()
            env["MUSICGENRE_TUNING_MODE"] = "1"
            print(f"\n=== {model_name.upper()} trial {trial}/{len(CANDIDATES[model_name])}: {config} ===", flush=True)
            subprocess.run([sys.executable, "-m", f"src.train_{model_name}"],
                           cwd=ROOT, env=env, check=True)
            result = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows.append({"model": model_name.upper(), "trial": trial,
                         "track_validation_accuracy": result["track_validation_accuracy"],
                         "macro_f1": result["macro_f1"], "config": json.dumps(config, sort_keys=True),
                         "search_epochs": args.epochs, "search_patience": args.patience,
                         "crops_per_track": args.crops_per_track,
                         "artifact_dir": run_dir.as_posix()})

    summary_path = REPORT_DIR / "architecture_tuning_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    best = {}
    for row in rows:
        model = row["model"]
        if model not in best or (row["macro_f1"], row["track_validation_accuracy"]) > (
                best[model]["macro_f1"], best[model]["track_validation_accuracy"]):
            best[model] = {**row, "config": json.loads(row["config"])}
    best_path = REPORT_DIR / "best_architectures.json"
    best_path.write_text(json.dumps(best, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(11, 6))
    for model in args.models:
        part = [row for row in rows if row["model"] == model.upper()]
        ax.plot([row["trial"] for row in part], [row["macro_f1"] for row in part],
                marker="o", label=model.upper())
    ax.set_xticks(range(1, max(row["trial"] for row in rows) + 1))
    ax.set_xlabel("Candidate configuration (see architecture_tuning_summary.csv)")
    ax.set_ylabel("Track-level validation macro F1")
    ax.set_title("Architecture and regularization search")
    ax.set_ylim(0, 1)
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "architecture_tuning_validation_f1.png", dpi=180)
    plt.close(fig)
    print(f"\nCandidate results: {summary_path}")
    print(f"Best validation configurations: {best_path}")
    print("Apply these settings for final training; do not use test results to choose them.")


if __name__ == "__main__":
    main()
