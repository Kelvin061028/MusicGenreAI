"""Run a small validation-only learning-rate sweep for the four models.

Example:
    python -m src.tune_models
    python -m src.tune_models --models cnn crnn --learning-rates 0.0001 0.0005 0.001

Each trial writes its checkpoint and history to separate tuning subfolders.
The test split is deliberately not evaluated during a tuning run.
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
REPORT_DIR = ROOT / "reports"
MODELS = ("cnn", "crnn", "lstm", "vit")


def lr_label(value: float) -> str:
    return f"{value:.0e}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument(
        "--learning-rates", nargs="+", type=float,
        default=[1e-4, 5e-4, 1e-3],
        help="Candidate Adam learning rates; defaults to 1e-4 5e-4 1e-3.",
    )
    args = parser.parse_args()
    if any(rate <= 0 for rate in args.learning_rates):
        parser.error("All learning rates must be positive.")

    tuning_root = REPORT_DIR / "tuning"
    tuning_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for model_name in args.models:
        for rate in args.learning_rates:
            run_dir = Path("tuning") / model_name / f"lr_{lr_label(rate)}"
            report_path = REPORT_DIR / run_dir / "validation_metrics.json"
            env = os.environ.copy()
            env["MUSICGENRE_LEARNING_RATE"] = str(rate)
            env["MUSICGENRE_RUN_DIR"] = run_dir.as_posix()
            env["MUSICGENRE_TUNING_MODE"] = "1"

            print(f"\n=== {model_name.upper()} | learning rate {rate:g} ===", flush=True)
            subprocess.run(
                [sys.executable, "-m", f"src.train_{model_name}"],
                cwd=ROOT,
                env=env,
                check=True,
            )

            with report_path.open("r", encoding="utf-8") as stream:
                metrics = json.load(stream)
            rows.append({
                "model": model_name.upper(),
                "learning_rate": rate,
                "track_validation_accuracy": metrics["track_validation_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "artifact_dir": run_dir.as_posix(),
            })

    summary_path = tuning_root / "tuning_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best_by_model: dict[str, dict[str, object]] = {}
    for row in rows:
        model_name = str(row["model"])
        previous = best_by_model.get(model_name)
        if previous is None or (
            float(row["macro_f1"]), float(row["track_validation_accuracy"])
        ) > (
            float(previous["macro_f1"]), float(previous["track_validation_accuracy"])
        ):
            best_by_model[model_name] = row

    best_path = tuning_root / "best_configurations.json"
    best_path.write_text(json.dumps(best_by_model, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name in args.models:
        model_rows = sorted(
            (row for row in rows if row["model"] == model_name.upper()),
            key=lambda row: float(row["learning_rate"]),
        )
        ax.plot(
            [float(row["learning_rate"]) for row in model_rows],
            [float(row["macro_f1"]) for row in model_rows],
            marker="o",
            label=model_name.upper(),
        )
    ax.set_xscale("log")
    ax.set_xticks(sorted(set(args.learning_rates)))
    ax.set_xticklabels([f"{rate:g}" for rate in sorted(set(args.learning_rates))])
    ax.set_xlabel("Adam learning rate")
    ax.set_ylabel("Track-level validation macro F1")
    ax.set_title("Learning-rate tuning by model")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(tuning_root / "validation_macro_f1_by_learning_rate.png", dpi=180)
    plt.close(fig)

    print(f"\nTuning summary: {summary_path}")
    print(f"Best validation configurations: {best_path}")
    print("Use the validation-selected rates for final runs, then evaluate the test split once.")


if __name__ == "__main__":
    main()
