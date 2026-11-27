"""Five-page Streamlit interface for MusicGenre AI."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


REPORT_FILES = {
    "CNN": "cnn_test_metrics.json",
    "CRNN": "crnn_test_metrics.json",
    "LSTM": "lstm_test_metrics.json",
    "Vision Transformer": "vision_transformer_test_metrics.json",
    "CNN + CRNN Ensemble": "ensemble_test_metrics.json",
}
TRAINING_HISTORY_FILES = {
    "CNN": "cnn_training_history.csv",
    "CRNN": "crnn_training_history.csv",
    "LSTM": "lstm_training_history.csv",
    "Vision Transformer": "vit_training_history.csv",
}
DESCRIPTIONS = {
    "CNN": "Learns local patterns in the log-Mel spectrogram.",
    "CRNN": "Combines convolutional feature extraction with bidirectional LSTM layers.",
    "LSTM": "Learns temporal patterns across log-Mel frames.",
    "Vision Transformer": "Uses self-attention across patches of the log-Mel spectrogram.",
    "CNN + CRNN Ensemble": "Equal-weight average of the CNN and CRNN genre probabilities.",
}
ARCHITECTURE_NOTES = {
    "CNN": "Four convolutional blocks extract spectrogram patterns before global pooling and classification.",
    "CRNN": "Four convolutional blocks form a time sequence that is passed through two bidirectional LSTM layers.",
    "LSTM": "Time pooling shortens each Mel-frame sequence before two LSTM layers classify it.",
    "Vision Transformer": "Each segment is resized to 128 × 128, split into 16 × 16 patches, then processed by transformer blocks.",
    "CNN + CRNN Ensemble": "For each genre, the ensemble takes the mean of the CNN and CRNN track-level scores.",
}


def _load_reports(root: Path):
    reports = {}
    for model, filename in REPORT_FILES.items():
        path = root / "reports" / filename
        if path.exists():
            try:
                reports[model] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
    return reports


def _summary(reports: dict):
    rows = []
    for model, report in reports.items():
        detail = report.get("classification_report", {})
        rows.append({
            "Model": model,
            "Accuracy": float(report.get("test_accuracy", 0)),
            "Macro F1": float(detail.get("macro avg", {}).get("f1-score", 0)),
            "Weighted F1": float(detail.get("weighted avg", {}).get("f1-score", 0)),
            "Test tracks": int(report.get("test_tracks", 0)),
        })
    return pd.DataFrame(rows)


def _prediction_table(st, result: dict, compact=False):
    predictions = result.get("predictions", {}) if result else {}
    if not predictions:
        st.info("Analyze a track in Music Analyzer to see its prediction results here.")
        return
    genres = result["genres"]
    rows, probabilities = [], {}
    for model, scores in predictions.items():
        scores = np.asarray(scores, dtype=float)
        order = np.argsort(scores)[::-1]
        rows.append({
            "Model": model,
            "Predicted genre": genres[int(order[0])].title(),
            "Score": float(scores[order[0]]),
            "Runner-up": genres[int(order[1])].title(),
        })
        probabilities[model] = scores
    table = pd.DataFrame(rows)
    table["Score"] = table["Score"].map(lambda value: f"{value:.1%}")
    st.dataframe(table, hide_index=True, use_container_width=True)
    if compact:
        return
    st.subheader("Genre score comparison")
    chart_data = pd.DataFrame(probabilities, index=[genre.title() for genre in genres])
    chart_data.index.name = "Genre"
    st.bar_chart(chart_data, use_container_width=True)
    with st.expander("Top three genres by model"):
        top_rows = []
        for model, scores in probabilities.items():
            for rank, index in enumerate(np.argsort(scores)[::-1][:3], start=1):
                top_rows.append({"Model": model, "Rank": rank,
                                 "Genre": genres[int(index)].title(),
                                 "Score": float(scores[index])})
        st.dataframe(pd.DataFrame(top_rows).style.format({"Score": "{:.1%}"}),
                     hide_index=True, use_container_width=True)
    st.caption("Scores average the model’s predictions across 3-second segments; they are not guarantees of correctness.")


def _music_characteristics(st, features):
    if not features:
        return
    st.divider()
    st.subheader("Music Characteristics")
    st.caption("Measured from the uploaded audio window. These descriptors are separate from the classifiers’ log-Mel inputs.")
    tempo = float(features.get("tempo_bpm", 0))
    metric_cols = st.columns(3)
    metric_cols[0].metric("Estimated tempo", f"{tempo:.0f} BPM" if tempo > 0 else "No clear beat")
    metric_cols[1].metric("RMS energy", f"{float(features.get('rms_energy', 0)):.3f}")
    metric_cols[2].metric("Spectral centroid", f"{float(features.get('spectral_centroid_hz', 0)) / 1000:.2f} kHz")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Spectral bandwidth", f"{float(features.get('spectral_bandwidth_hz', 0)) / 1000:.2f} kHz")
    metric_cols[1].metric("Spectral rolloff", f"{float(features.get('spectral_rolloff_hz', 0)) / 1000:.2f} kHz")
    metric_cols[2].metric("Zero-crossing rate", f"{float(features.get('zero_crossing_rate', 0)):.3f}")
    chroma = np.asarray(features.get("chroma", []), dtype=float)
    if chroma.size == 12:
        pitch_classes = ["C", "C♯", "D", "D♯", "E", "F", "F♯", "G", "G♯", "A", "A♯", "B"]
        st.markdown("**Chroma distribution**")
        chroma_frame = pd.DataFrame({"Mean intensity": chroma}, index=pitch_classes)
        chroma_frame.index.name = "Pitch class"
        st.bar_chart(chroma_frame, use_container_width=True)
    with st.expander("What the measurements mean"):
        st.markdown(
            "- **Tempo:** estimated beats per minute.\n"
            "- **RMS energy:** average signal amplitude.\n"
            "- **Spectral centroid:** a brightness-related measure.\n"
            "- **Bandwidth:** frequency spread around the centroid.\n"
            "- **Rolloff:** frequency below which most spectral energy lies.\n"
            "- **Zero-crossing rate:** how often the waveform crosses zero.\n"
            "- **Chroma:** relative energy across the 12 pitch classes."
        )


def _automatic_interpretation(st, features, predictions, genres):
    """Summarize measured descriptors and the classifier's top label cautiously."""
    if not features or not predictions:
        return

    # Prefer the ensemble when available; otherwise use the first selected model.
    model_name = "CNN + CRNN Ensemble" if "CNN + CRNN Ensemble" in predictions else next(iter(predictions))
    scores = np.asarray(predictions[model_name], dtype=float)
    if not scores.size or len(genres) != scores.size:
        return
    predicted_genre = str(genres[int(np.argmax(scores))]).title()

    tempo = float(features.get("tempo_bpm", 0))
    if tempo <= 0:
        tempo_text = "no clear tempo estimate"
    else:
        pace = "brisk" if tempo >= 120 else "moderate" if tempo >= 90 else "relaxed"
        tempo_text = f"an estimated tempo of {tempo:.0f} BPM ({pace})"

    energy = float(features.get("rms_energy", 0))
    energy_text = "higher" if energy >= 0.15 else "moderate" if energy >= 0.07 else "lower"
    centroid = float(features.get("spectral_centroid_hz", 0))
    brightness = "brighter" if centroid >= 2500 else "less bright" if centroid < 1500 else "moderately bright"

    st.subheader("Automatic interpretation")
    st.write(
        f"The uploaded track has {tempo_text}, {energy_text} RMS energy and a "
        f"{brightness} spectral profile. Its top predicted genre is **{predicted_genre}** "
        f"({model_name})."
    )
    st.caption(
        "These are descriptive audio measurements and a model prediction; the measurements "
        "do not prove the genre. Descriptive bands use simple rules."
    )


def _dashboard(st, reports, context):
    st.title("Dashboard")
    st.caption("A quick overview of model availability and this session's activity.")
    summary = _summary(reports)
    latest = st.session_state.get("latest_prediction")
    history_count = len(st.session_state.get("prediction_history", []))
    best = summary.sort_values("Accuracy", ascending=False).iloc[0] if not summary.empty else None
    cols = st.columns(4)
    cols[0].metric("Classifiers", "4")
    cols[1].metric("Genres", str(len(context["genres"])))
    cols[2].metric("Best test accuracy", f"{best['Accuracy']:.1%}" if best is not None else "—",
                   best["Model"] if best is not None else "No report")
    cols[3].metric("Session predictions", str(history_count))
    if not summary.empty:
        st.subheader("Model comparison summary")
        st.bar_chart(summary.set_index("Model")[["Accuracy", "Macro F1"]],
                     use_container_width=True)
        compact_summary = summary[["Model", "Accuracy", "Macro F1"]]
        st.dataframe(compact_summary.style.format({"Accuracy": "{:.1%}", "Macro F1": "{:.3f}"}),
                     hide_index=True, use_container_width=True)
    st.subheader("Latest activity")
    if latest:
        predictions = latest.get("predictions", {})
        ensemble = predictions.get("CNN + CRNN Ensemble")
        if ensemble is not None:
            genres = latest.get("genres", context["genres"])
            top = int(np.argmax(ensemble))
            st.info(f"Latest track: **{latest['track']}** · ensemble prediction: **{genres[top].title()}** · {latest['time']}")
        else:
            st.info(f"Latest track: **{latest['track']}** · {len(predictions)} model results · {latest['time']}")
    else:
        st.info("No audio has been analyzed in this session yet.")
    if latest and latest.get("errors"):
        st.caption("Some models were unavailable: " + ", ".join(latest["errors"]))
    if summary.empty:
        st.caption("No evaluation reports found. Run `python -m src.evaluate_final` to create them.")

def _music_analyzer(st, context):
    st.title("Music Analyzer")
    st.caption("Upload a track and compare predictions from the selected models.")
    uploaded = st.file_uploader(
        "Choose an audio file", type=["wav", "mp3", "ogg", "flac"],
        help=f"Predictions use up to the first {context['duration']} seconds.",
        key="music_analyzer_upload",
    )
    if uploaded is not None:
        st.audio(uploaded.getvalue(), format=uploaded.type or None)

    with st.form("music_analyzer_form"):
        selected_models = st.multiselect(
            "Models to run", list(context["model_files"]),
            default=list(context["model_files"]),
            help="The CNN + CRNN ensemble is added when both models are selected.",
        )
        submitted = st.form_submit_button("Analyze track", type="primary", use_container_width=True)

    if submitted:
        if uploaded is None:
            st.error("Choose an audio file first.")
        elif not selected_models:
            st.error("Select at least one model.")
        elif not context["norm_stats_path"].exists():
            st.error("Training normalization statistics are missing.")
        else:
            audio_bytes = uploaded.getvalue()
            stats = json.loads(context["norm_stats_path"].read_text(encoding="utf-8"))
            genre_names = stats.get("class_names", context["genres"])
            model_versions = {
                name: (context["model_dir"] / filename).stat().st_mtime_ns
                for name, filename in context["model_files"].items()
                if name in selected_models and (context["model_dir"] / filename).exists()
            }
            with st.spinner("Analyzing audio with the selected classifiers..."):
                try:
                    predictions, errors, music_characteristics = context["predict"](
                        audio_bytes, uploaded.name, model_versions,
                        json.dumps(stats, sort_keys=True), genre_names, selected_models,
                    )
                except Exception as error:
                    predictions, errors, music_characteristics = {}, {"Audio processing": str(error)}, None
            for model_name, message in errors.items():
                st.warning(f"{model_name}: {message}")
            if predictions:
                now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
                signature = context["signature"](audio_bytes, selected_models)
                latest = {
                    "track": uploaded.name, "time": now, "genres": genre_names,
                    "predictions": {name: np.asarray(scores).tolist()
                                    for name, scores in predictions.items()},
                    "errors": errors, "signature": signature,
                    "music_characteristics": music_characteristics,
                }
                st.session_state.latest_prediction = latest
                if st.session_state.get("last_prediction_signature") != signature:
                    st.session_state.setdefault("prediction_history", []).insert(0, {
                        "time": now, "track": uploaded.name, "models": list(predictions),
                        "music_characteristics": music_characteristics,
                        "predictions": {
                            name: {"genre": genre_names[int(np.argmax(scores))],
                                   "score": float(np.max(scores)),
                                   "probabilities": np.asarray(scores, dtype=float).tolist()}
                            for name, scores in predictions.items()
                        },
                    })
                    st.session_state.prediction_history = st.session_state.prediction_history[:20]
                    st.session_state.last_prediction_signature = signature

    latest = st.session_state.get("latest_prediction")
    if latest:
        st.divider()
        st.subheader(f"Results for {latest['track']}")
        st.caption(f"Analyzed {latest['time']} · inference uses up to {context['duration']} seconds.")
        _prediction_table(st, latest)
        _automatic_interpretation(
            st, latest.get("music_characteristics"), latest.get("predictions", {}),
            latest.get("genres", context["genres"]),
        )
        _music_characteristics(st, latest.get("music_characteristics"))
    else:
        st.info("Your model comparison will appear here after you analyze a track.")


def _model_laboratory(st, context):
    st.title("Model Laboratory")
    st.caption("Inspect the four classifiers, their selected settings, and how the ensemble works.")
    names = list(context["model_files"]) + ["CNN + CRNN Ensemble"]
    selected = st.selectbox("Choose a model", names)
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader(selected)
        st.write(DESCRIPTIONS[selected])
        if selected == "CNN + CRNN Ensemble":
            st.markdown("**Method:** average the CNN and CRNN probability vectors with equal weights.")
            st.markdown("**Output:** a genre ranking for the uploaded track.")
        else:
            checkpoint = context["model_dir"] / context["model_files"][selected]
            st.markdown(f"**Checkpoint:** {'Available' if checkpoint.exists() else 'Missing'}")
            st.markdown(f"**Input:** 128-bin log-Mel segments · up to {context['duration']} seconds")
            best_path = context["root"] / "reports" / "tuning" / "best_architectures.json"
            if best_path.exists():
                try:
                    configs = json.loads(best_path.read_text(encoding="utf-8"))
                    key = "VIT" if selected == "Vision Transformer" else selected.upper()
                    entry = configs.get(key)
                    if entry:
                        st.markdown("**Validation-selected settings**")
                        st.json(entry.get("config", {}), expanded=True)
                        st.caption(f"Selected from: {entry.get('selection_source', 'architecture search')}")
                except (OSError, json.JSONDecodeError):
                    st.caption("The tuning configuration report could not be read.")
    with right:
        st.subheader("Architecture")
        st.write(ARCHITECTURE_NOTES[selected])
        st.markdown("**Shared prediction flow**")
        st.caption("Log-Mel spectrogram → 3-second segments → genre probabilities → track-level average")
        if selected == "CNN + CRNN Ensemble":
            st.caption("Both component models must be available for the ensemble prediction.")

def _performance_analysis(st, reports, root: Path):
    st.title("Performance Analysis")
    st.caption("Held-out test metrics, per-genre performance, and training diagnostics.")
    summary = _summary(reports)
    if summary.empty:
        st.info("No evaluation reports found. Run `python -m src.evaluate_final` first.")
        return
    selected = st.selectbox("Inspect a model", list(reports), key="performance_model")
    selected_row = summary[summary["Model"] == selected]
    if not selected_row.empty:
        item = selected_row.iloc[0]
        cols = st.columns(4)
        cols[0].metric("Accuracy", f"{item['Accuracy']:.1%}")
        cols[1].metric("Macro F1", f"{item['Macro F1']:.3f}")
        cols[2].metric("Weighted F1", f"{item['Weighted F1']:.3f}")
        cols[3].metric("Test tracks", str(item["Test tracks"]))

    report = reports[selected]
    details = report.get("classification_report", {})
    genre_rows = []
    for genre in report.get("class_names", []):
        item = details.get(genre, {})
        if item:
            genre_rows.append({"Genre": genre.title(), "Precision": item.get("precision", 0),
                               "Recall": item.get("recall", 0), "F1": item.get("f1-score", 0),
                               "Support": int(item.get("support", 0))})
    if genre_rows:
        genre_frame = pd.DataFrame(genre_rows)
        st.subheader(f"Per-genre results · {selected}")
        st.bar_chart(genre_frame.set_index("Genre")[["Precision", "Recall", "F1"]],
                     use_container_width=True)
        st.dataframe(genre_frame.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"}),
                     hide_index=True, use_container_width=True)

    slug = "ensemble" if selected == "CNN + CRNN Ensemble" else selected.lower().replace(" ", "_")
    matrix_path = root / "reports" / f"{slug}_final_test_confusion_matrix.png"
    if matrix_path.exists():
        st.subheader("Confusion matrix")
        st.image(str(matrix_path), use_container_width=True)
    history_file = TRAINING_HISTORY_FILES.get(selected)
    history_path = root / "reports" / history_file if history_file else None
    if history_path and history_path.exists():
        history = pd.read_csv(history_path)
        columns = [key for key in ("accuracy", "val_accuracy", "loss", "val_loss") if key in history]
        if columns:
            st.subheader("Training history")
            st.line_chart(history.set_index("epoch")[columns], use_container_width=True)
            st.caption("Training/validation curves are segment-level; reported test metrics aggregate by track.")

def _history_and_reports(st, reports):
    st.title("History and Reports")
    st.caption("Prediction history is kept in this browser session and can be exported.")
    history = st.session_state.get("prediction_history", [])
    if history:
        rows = []
        for entry in history:
            row = {"Time": entry["time"], "Track": entry["track"], "Models": ", ".join(entry["models"])}
            for model, item in entry["predictions"].items():
                row[model] = f"{item['genre'].title()} ({item['score']:.1%})"
            features = entry.get("music_characteristics") or {}
            if features:
                row["Tempo BPM"] = features.get("tempo_bpm")
                row["RMS Energy"] = features.get("rms_energy")
                row["Spectral Centroid Hz"] = features.get("spectral_centroid_hz")
                row["Spectral Bandwidth Hz"] = features.get("spectral_bandwidth_hz")
                row["Spectral Rolloff Hz"] = features.get("spectral_rolloff_hz")
                row["Zero-Crossing Rate"] = features.get("zero_crossing_rate")
                row["Chroma"] = json.dumps(features.get("chroma", []))
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.download_button("Download history (JSON)", json.dumps(history, indent=2).encode("utf-8"),
                           file_name="musicgenre_prediction_history.json", mime="application/json")
        st.download_button("Download history (CSV)", pd.DataFrame(rows).to_csv(index=False).encode("utf-8"),
                           file_name="musicgenre_prediction_history.csv", mime="text/csv")
        if st.button("Clear session history"):
            st.session_state.prediction_history = []
            st.session_state.pop("last_prediction_signature", None)
            st.rerun()
    else:
        st.info("No predictions have been saved this session.")
    st.divider()
    st.subheader("Final evaluation reports")
    summary = _summary(reports)
    if not summary.empty:
        st.download_button("Download model comparison (CSV)", summary.to_csv(index=False).encode("utf-8"),
                           file_name="musicgenre_test_comparison.csv", mime="text/csv")
        model = st.selectbox("Detailed report", list(reports), key="detailed_report_select")
        st.download_button(f"Download {model} report (JSON)", json.dumps(reports[model], indent=2).encode("utf-8"),
                           file_name=f"{model.lower().replace(' ', '_')}_test_report.json",
                           mime="application/json")


def run_pages(context):
    st = context["st"]
    st.set_page_config(page_title="MusicGenre AI", page_icon="🎵", layout="wide")
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []
    with st.sidebar:
        st.markdown("## 🎵 MusicGenre AI")
        st.caption("AI-powered music analysis platform")
        page = st.radio(
            "Navigate",
            ["Dashboard", "Music Analyzer", "Model Laboratory", "Performance Analysis", "History and Reports"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("GTZAN · four classifiers · CNN + CRNN ensemble")

    reports = _load_reports(context["root"])
    if page == "Dashboard":
        _dashboard(st, reports, context)
    elif page == "Music Analyzer":
        _music_analyzer(st, context)
    elif page == "Model Laboratory":
        _model_laboratory(st, context)
    elif page == "Performance Analysis":
        _performance_analysis(st, reports, context["root"])
    else:
        _history_and_reports(st, reports)
    st.divider()
    st.caption("MusicGenre AI · GTZAN genre classification · inference uses up to 30 seconds of audio")
