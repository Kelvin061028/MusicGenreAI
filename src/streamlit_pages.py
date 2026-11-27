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


def _format_summary(st, frame):
    if frame.empty:
        return
    st.dataframe(
        frame.style.format({"Accuracy": "{:.1%}", "Macro F1": "{:.3f}",
                            "Weighted F1": "{:.3f}"}),
        hide_index=True, use_container_width=True,
    )


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


def _dashboard(st, reports, context):
    st.title("Dashboard")
    st.caption("Overview of model results and predictions made in this session.")
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

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Held-out model comparison")
        if summary.empty:
            st.info("No evaluation reports found. Run `python -m src.evaluate_final` to create them.")
        else:
            st.bar_chart(summary.set_index("Model")[["Accuracy", "Macro F1"]],
                         use_container_width=True)
    with right:
        st.subheader("Latest uploaded-track result")
        if latest:
            st.caption(f"{latest['track']} · {latest['time']}")
            _prediction_table(st, latest, compact=True)
        else:
            st.info("No audio has been analyzed in this session yet.")
    if not summary.empty:
        with st.expander("Evaluation summary"):
            _format_summary(st, summary)


def _music_analyzer(st, context):
    st.title("Music Analyzer")
    st.caption("Upload a track and compare predictions from the selected models.")
    with st.form("music_analyzer_form"):
        uploaded = st.file_uploader(
            "Choose an audio file", type=["wav", "mp3", "ogg", "flac"],
            help=f"Predictions use up to the first {context['duration']} seconds.",
        )
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
                    predictions, errors = context["predict"](
                        audio_bytes, uploaded.name, model_versions,
                        json.dumps(stats, sort_keys=True), genre_names, selected_models,
                    )
                except Exception as error:
                    predictions, errors = {}, {"Audio processing": str(error)}
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
                }
                st.session_state.latest_prediction = latest
                if st.session_state.get("last_prediction_signature") != signature:
                    st.session_state.setdefault("prediction_history", []).insert(0, {
                        "time": now, "track": uploaded.name, "models": list(predictions),
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
    else:
        st.info("Your model comparison will appear here after you analyze a track.")


def _model_laboratory(st, reports, context):
    st.title("Model Laboratory")
    st.caption("Inspect the four classifiers, their selected settings, and the ensemble.")
    names = list(context["model_files"]) + ["CNN + CRNN Ensemble"]
    selected = st.selectbox("Choose a model", names)
    summary = _summary(reports)
    left, right = st.columns([1.1, 1])
    with left:
        st.subheader(selected)
        st.write(DESCRIPTIONS[selected])
        if selected == "CNN + CRNN Ensemble":
            st.markdown("**Method:** equal-weight average of CNN and CRNN probability scores.")
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
        st.subheader("Held-out test metrics")
        row = summary[summary["Model"] == selected] if not summary.empty else pd.DataFrame()
        if row.empty:
            st.info("No test report was found for this model.")
        else:
            item = row.iloc[0]
            st.metric("Accuracy", f"{item['Accuracy']:.1%}")
            st.metric("Macro F1", f"{item['Macro F1']:.3f}")
            st.metric("Weighted F1", f"{item['Weighted F1']:.3f}")
            st.caption(f"Test tracks: {item['Test tracks']}")
    latest = st.session_state.get("latest_prediction")
    if latest:
        st.divider()
        st.subheader("Latest uploaded-track comparison")
        _prediction_table(st, latest, compact=True)
    st.divider()
    st.subheader("All models and ensemble")
    _format_summary(st, summary)


def _performance_analysis(st, reports, root: Path):
    st.title("Performance Analysis")
    st.caption("Held-out test metrics, per-genre scores, confusion matrices, and training curves.")
    summary = _summary(reports)
    if summary.empty:
        st.info("No evaluation reports found. Run `python -m src.evaluate_final` first.")
        return
    st.subheader("Model-level performance")
    st.bar_chart(summary.set_index("Model")[["Accuracy", "Macro F1", "Weighted F1"]],
                 use_container_width=True)
    _format_summary(st, summary)

    selected = st.selectbox("Inspect a model", list(reports), key="performance_model")
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
    _format_summary(st, summary)
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
        _model_laboratory(st, reports, context)
    elif page == "Performance Analysis":
        _performance_analysis(st, reports, context["root"])
    else:
        _history_and_reports(st, reports)
    st.divider()
    st.caption("MusicGenre AI · GTZAN genre classification · inference uses up to 30 seconds of audio")
