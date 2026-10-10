"""Interactive Hinglish sentiment analysis demo."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import available_models, load_predictor, predict_text

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
MAX_INPUT_CHARS = 2_000
EXAMPLES = [
    "Movie ekdum mast thi, paisa vasool!",
    "Yaar service bahut slow thi, mood kharab ho gaya.",
    "Product theek hai, kuch special nahi.",
    "Camera quality zabardast hai bhai.",
    "App baar baar crash ho raha hai, very bad.",
    "Food tasty tha but delivery late aayi.",
    "Wah kya amazing support hai, thank you!",
    "Phone ka design simple hai aur battery normal hai.",
    "Itna bhi bura nahi tha, but expected better.",
    "Bhai ye update bilkul useless hai.",
]

st.set_page_config(page_title="Hinglish Sentiment Analysis", page_icon="H", layout="centered")
st.title("Hinglish Sentiment Analysis")
st.caption("Positive / negative / neutral sentiment for Hindi-English code-mixed text.")


@st.cache_resource(show_spinner="Loading selected model…")
def cached_predictor(name: str) -> dict:
    """Cache model memory across Streamlit reruns."""
    return load_predictor(name, CONFIG_PATH)


def render_prediction(text: str, predictor: dict) -> None:
    """Display one prediction and a readable confidence chart."""
    if len(text) > MAX_INPUT_CHARS:
        st.info(f"Input was limited to the first {MAX_INPUT_CHARS:,} characters.")
        text = text[:MAX_INPUT_CHARS]
    try:
        result = predict_text(text, predictor, CONFIG_PATH)
    except ValueError as error:
        st.warning(str(error))
        return
    st.subheader(f"Prediction: {result.label.title()}")
    st.progress(int(result.confidence * 100), text=f"Confidence: {result.confidence:.1%}")
    st.bar_chart(pd.DataFrame.from_dict(result.probabilities, orient="index", columns=["confidence"]))
    st.caption(f"Cleaned text: {result.cleaned_text}")


available = available_models(CONFIG_PATH)
if not available:
    st.error("No trained models were found. Train a baseline model first.")
    st.stop()

with st.sidebar:
    st.header("Model")
    selected = st.selectbox("Choose a trained model", list(available), format_func=lambda item: item.replace("_", " ").title())
    st.caption("Only locally available model files are listed.")
    st.divider()
    st.caption("Long comments are truncated to the selected model's configured maximum length.")

predictor = cached_predictor(selected)
text = st.text_area("Enter a Hinglish comment", height=120, placeholder="Movie ekdum bakwas thi, paisa waste")
if st.button("Analyze", type="primary"):
    render_prediction(text, predictor)

st.subheader("Try an example")
example_columns = st.columns(2)
for index, example in enumerate(EXAMPLES):
    with example_columns[index % 2]:
        if st.button(example, key=f"example_{index}", use_container_width=True):
            render_prediction(example, predictor)

st.divider()
st.subheader("Batch prediction")
uploaded = st.file_uploader("Upload a CSV with a text column", type="csv")
if uploaded is not None:
    try:
        batch = pd.read_csv(uploaded)
        if "text" not in batch.columns:
            raise ValueError("The uploaded CSV needs a column named text.")
        if len(batch) > 10_000:
            raise ValueError("Batch uploads are limited to 10,000 rows.")
        predictions = [predict_text(str(value)[:MAX_INPUT_CHARS], predictor, CONFIG_PATH) for value in batch["text"].fillna("")]
        batch["predicted_label"] = [item.label for item in predictions]
        batch["confidence"] = [item.confidence for item in predictions]
        batch["cleaned_text"] = [item.cleaned_text for item in predictions]
        st.dataframe(batch.head(20), use_container_width=True)
        st.download_button("Download predictions CSV", batch.to_csv(index=False).encode("utf-8"), "hinglish_sentiment_predictions.csv", "text/csv")
    except (ValueError, UnicodeDecodeError) as error:
        st.error(str(error))
