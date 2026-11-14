
from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"

MODELS = ["cnn", "crnn", "lstm", "vit"]

# ==========================================
# 2. LOAD MODEL METRICS
# ==========================================

results = []

for model_name in MODELS:

    file_path = (
        REPORT_DIR /
        f"{model_name}_test_metrics.json"
    )

    if not file_path.exists():
        print(f"[PENDING] {model_name.upper()} results not found.")
        continue

    with open(file_path, "r") as file:
        metrics = json.load(file)

    report = metrics.get(
        "classification_report", {}
    )

    macro = report.get("macro avg", {})
    weighted = report.get("weighted avg", {})

    results.append({
        "Model": model_name.upper(),
        "Test Loss": metrics.get("test_loss"),
        "Accuracy": metrics.get("test_accuracy"),
        "Precision": macro.get("precision"),
        "Recall": macro.get("recall"),
        "Macro F1": macro.get("f1-score"),
        "Weighted F1": weighted.get("f1-score"),
    })

# ==========================================
# 3. CREATE COMPARISON TABLE
# ==========================================

if not results:
    raise FileNotFoundError(
        "No model metrics found in reports folder."
    )

df = pd.DataFrame(results)

df = df.sort_values(
    by="Accuracy",
    ascending=False,
    na_position="last"
).reset_index(drop=True)

df.insert(
    0,
    "Rank",
    range(1, len(df) + 1)
)

print("\n========== MODEL COMPARISON ==========\n")

print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

# Save comparison
df.to_csv(
    REPORT_DIR / "model_comparison.csv",
    index=False
)

# ==========================================
# 4. ACCURACY COMPARISON
# ==========================================

plt.figure(figsize=(9, 5))

plt.bar(
    df["Model"],
    df["Accuracy"]
)

plt.title("Test Accuracy Comparison")
plt.xlabel("Deep Learning Architecture")
plt.ylabel("Test Accuracy")
plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "model_accuracy_comparison.png",
    dpi=300
)

plt.show()

# ==========================================
# 5. F1-SCORE COMPARISON
# ==========================================

plt.figure(figsize=(9, 5))

plt.bar(
    df["Model"],
    df["Macro F1"]
)

plt.title("Macro F1-Score Comparison")
plt.xlabel("Deep Learning Architecture")
plt.ylabel("Macro F1-Score")
plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "model_f1_comparison.png",
    dpi=300
)

plt.show()

# ==========================================
# 6. TRAINING HISTORY COMPARISON
# ==========================================

for metric in ["accuracy", "loss"]:

    plt.figure(figsize=(10, 6))

    found_history = False

    for model_name in MODELS:

        history_path = (
            REPORT_DIR /
            f"{model_name}_training_history.csv"
        )

        if not history_path.exists():
            continue

        history = pd.read_csv(history_path)

        train_column = metric
        val_column = f"val_{metric}"

        if train_column not in history.columns:
            continue

        plt.plot(
            history[train_column],
            label=f"{model_name.upper()} Training"
        )

        if val_column in history.columns:
            plt.plot(
                history[val_column],
                linestyle="--",
                label=f"{model_name.upper()} Validation"
            )

        found_history = True

    if found_history:

        plt.title(
            f"Model Training and Validation {metric.title()}"
        )

        plt.xlabel("Epoch")
        plt.ylabel(metric.title())
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(
            REPORT_DIR /
            f"model_{metric}_comparison.png",
            dpi=300
        )

        plt.show()

    else:
        plt.close()

print("\nComparison completed.")
print("Files saved in the reports folder.")
