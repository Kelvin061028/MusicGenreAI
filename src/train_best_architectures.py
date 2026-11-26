"""Retrain every model using reports/tuning/best_architectures.json, then evaluate."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "reports" / "tuning" / "best_architectures.json"
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
    parser.add_argument("--epochs", type=int, default=80,
                        help="Maximum epochs for final training (default: 80).")
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--models", nargs="+", choices=("cnn", "crnn", "vit", "lstm"),
                        default=["cnn", "crnn", "vit", "lstm"])
    args = parser.parse_args()
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Run src.tune_architectures first; missing {CONFIG_PATH}")
    configurations = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for model in args.models:
        entry = configurations.get(model.upper())
        if entry is None:
            raise KeyError(f"No selected validation configuration for {model.upper()} in {CONFIG_PATH}")
        env = os.environ.copy()
        for key, value in entry["config"].items():
            env[ENV_KEYS[key]] = str(value)
        env.pop("MUSICGENRE_RUN_DIR", None)
        env.pop("MUSICGENRE_TUNING_MODE", None)
        env["MUSICGENRE_CROPS_PER_TRACK"] = "6"
        env["MUSICGENRE_EPOCHS"] = str(args.epochs)
        env["MUSICGENRE_EARLY_STOPPING_PATIENCE"] = str(args.patience)
        env["MUSICGENRE_LR_PATIENCE"] = str(max(2, args.patience // 2))
        print(f"\n=== Final training {model.upper()} with {entry['config']} ===", flush=True)
        subprocess.run([sys.executable, "-m", f"src.train_{model}"],
                       cwd=ROOT, env=env, check=True)
    print("\nAll selected checkpoints trained. Running final held-out test evaluation.", flush=True)
    subprocess.run([sys.executable, "-m", "src.evaluate_final"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
