"""Streamlit application for AI-based waste classification and recovery recommendations."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

from recovery_engine import get_recovery_info

ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "waste_model.h5"
CLASS_NAMES_PATH = ROOT_DIR / "class_names.json"
HISTORY_PATH = ROOT_DIR / "history.csv"


def load_css() -> None:
    """Inject custom CSS for the UI."""
    st.markdown(
        """
        <style>
            :root {
                --bg-1: #071b1a;
                --bg-2: #0d2c2f;
                --panel: rgba(13, 29, 35, 0.78);
                --panel-strong: rgba(18, 38, 43, 0.92);
                --line: rgba(142, 255, 219, 0.2);
                --text: #edfdf7;
                --muted: #b9d9d0;
                --primary: #5eead4;
                --primary-2: #7dd3fc;
                --green: #34d399;
                --amber: #fbbf24;
                --danger: #fb7185;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(94, 234, 212, 0.15), transparent 28%),
                    radial-gradient(circle at bottom right, rgba(125, 211, 252, 0.15), transparent 24%),
                    linear-gradient(135deg, var(--bg-1) 0%, var(--bg-2) 38%, #081e22 100%);
                color: var(--text);
            }

            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 3rem;
            }

            [data-testid="stSidebar"] {
                background: rgba(8, 24, 27, 0.9);
                border-right: 1px solid var(--line);
                box-shadow: 10px 0 30px rgba(0, 0, 0, 0.18);
            }

            [data-testid="stSidebar"] .stRadio {
                padding-top: 1.2rem;
            }

            [data-testid="stSidebar"] .stRadio > div {
                gap: 0.6rem;
            }

            [data-testid="stSidebar"] label {
                border-radius: 14px;
                padding: 0.7rem 0.75rem;
                border: 1px solid transparent;
                background: rgba(255,255,255,0.02);
                transition: all 0.25s ease;
            }

            [data-testid="stSidebar"] label:hover {
                border-color: rgba(94, 234, 212, 0.35);
                background: rgba(94, 234, 212, 0.08);
                transform: translateX(2px);
            }

            [data-testid="stSidebar"] input[type="radio"] {
                accent-color: var(--primary);
            }

            h1, h2, h3, h4, p, li, div {
                color: var(--text);
            }

            .hero-panel {
                position: relative;
                overflow: hidden;
                padding: 2rem 2rem 1.5rem 2rem;
                margin-bottom: 1.5rem;
                border-radius: 26px;
                background: linear-gradient(135deg, rgba(15, 70, 66, 0.9), rgba(12, 35, 42, 0.85));
                border: 1px solid rgba(94, 234, 212, 0.25);
                box-shadow: 0 24px 55px rgba(2, 8, 20, 0.45);
            }

            .hero-panel::before {
                content: "";
                position: absolute;
                inset: -40% -20% auto auto;
                width: 240px;
                height: 240px;
                background: radial-gradient(circle, rgba(94, 234, 212, 0.35), rgba(94, 234, 212, 0));
                filter: blur(18px);
                pointer-events: none;
            }

            .hero-kicker {
                display: inline-block;
                padding: 0.4rem 0.8rem;
                border-radius: 999px;
                background: rgba(94, 234, 212, 0.12);
                border: 1px solid rgba(94, 234, 212, 0.28);
                letter-spacing: 0.12em;
                text-transform: uppercase;
                font-size: 0.7rem;
                color: var(--primary);
                font-weight: 700;
            }

            .big-font {
                margin-top: 0.9rem;
                font-size: clamp(2.2rem, 5vw, 3.6rem);
                font-weight: 800;
                line-height: 1.1;
                color: #f6fffd;
            }

            .sub-head {
                margin-top: 0.8rem;
                color: var(--muted);
                font-size: 1.05rem;
                max-width: 700px;
            }

            .feature-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 1rem;
                margin-top: 1.5rem;
            }

            .metric-card {
                position: relative;
                background: linear-gradient(135deg, rgba(18, 38, 43, 0.9), rgba(9, 24, 28, 0.8));
                border: 1px solid rgba(125, 211, 252, 0.14);
                border-radius: 20px;
                padding: 1.2rem;
                box-shadow: 0 16px 35px rgba(0, 0, 0, 0.18);
                transition: transform 0.25s ease, border-color 0.25s ease;
            }

            .metric-card:hover {
                transform: translateY(-4px);
                border-color: rgba(94, 234, 212, 0.3);
            }

            .metric-card h3 {
                margin-bottom: 0.5rem;
                font-size: 1.08rem;
            }

            .metric-card p {
                color: var(--muted);
                margin-bottom: 0;
            }

            .premium-card {
                background: linear-gradient(180deg, rgba(14, 30, 34, 0.9), rgba(11, 24, 27, 0.92));
                border: 1px solid var(--line);
                border-radius: 22px;
                padding: 1.4rem;
                box-shadow: 0 18px 45px rgba(0, 0, 0, 0.2);
                margin-top: 1rem;
            }

            .premium-card h3 {
                margin-top: 0;
            }

            .basic-light {
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid var(--line);
                border-radius: 16px;
                padding: 0.8rem 1rem;
            }

            .stButton > button {
                background: linear-gradient(135deg, var(--primary), var(--primary-2));
                color: #021b1f;
                font-weight: 700;
                border: none;
                border-radius: 12px;
                padding: 0.7rem 1.25rem;
                box-shadow: 0 12px 30px rgba(94, 234, 212, 0.35);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }

            .stButton > button:hover {
                transform: translateY(-1px) scale(1.01);
                box-shadow: 0 18px 35px rgba(94, 234, 212, 0.45);
            }

            .stFileUploader > div {
                background: rgba(255,255,255,0.02);
                border: 1px dashed rgba(94, 234, 212, 0.35);
                border-radius: 18px;
                box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
            }

            .stFileUploader label {
                color: var(--text) !important;
            }

            .stAlert {
                border-radius: 16px;
                border: 1px solid rgba(125, 211, 252, 0.35);
                background: rgba(13, 29, 35, 0.78);
            }

            .stProgress > div > div {
                background: linear-gradient(90deg, var(--primary), var(--green));
            }

            .stDataFrame {
                border-radius: 18px;
                overflow: hidden;
                border: 1px solid rgba(94, 234, 212, 0.15);
            }

            .stMetric {
                background: rgba(9, 24, 28, 0.7);
                border: 1px solid rgba(94, 234, 212, 0.12);
                border-radius: 18px;
                padding: 1rem;
                box-shadow: 0 12px 30px rgba(0,0,0,0.12);
            }

            .stMetric [data-testid="stMetricValue"] {
                color: #f4fffb;
            }

            @media (max-width: 900px) {
                .feature-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model_and_classes():
    """Load the trained model and class labels from disk."""
    model_file = MODEL_PATH if MODEL_PATH.exists() else BEST_MODEL_PATH
    if not model_file.exists() or not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Please run 'python train_model.py' first to train the model."
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    model = tf.keras.models.load_model(str(model_file), compile=False)
    return model, class_names


def preprocess_image(img: Image.Image, target_size: tuple[int, int] = (224, 224)):
    """Convert and resize a PIL image into the exact format used by the model."""
    img_resized = img.convert("RGB").resize(target_size, Image.Resampling.BILINEAR)
    image_array = np.array(img_resized, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def log_prediction(class_name: str, confidence: float, weight_kg: float, value_inr: float, co2_saved_kg: float):
    """Append the prediction to the history.csv file."""
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "class": str(class_name),
        "confidence": round(float(confidence), 4),
        "weight_kg": round(float(weight_kg), 2),
        "value_inr": round(float(value_inr), 2),
        "co2_saved_kg": round(float(co2_saved_kg), 2),
    }

    cols = ["timestamp", "class", "confidence", "weight_kg", "value_inr", "co2_saved_kg"]
    if HISTORY_PATH.exists():
        try:
            history_df = pd.read_csv(HISTORY_PATH)
            history_df = pd.concat([history_df, pd.DataFrame([payload])], ignore_index=True)
        except Exception:
            history_df = pd.DataFrame([payload], columns=cols)
    else:
        history_df = pd.DataFrame([payload], columns=cols)

    history_df.to_csv(HISTORY_PATH, index=False)


def summarize_dashboard() -> pd.DataFrame:
    """Load the prediction history for dashboard metrics."""
    default_cols = ["timestamp", "class", "confidence", "weight_kg", "value_inr", "co2_saved_kg"]
    if not HISTORY_PATH.exists():
        return pd.DataFrame(columns=default_cols)
    try:
        df = pd.read_csv(HISTORY_PATH)
        for col in default_cols:
            if col not in df.columns:
                df[col] = 0.0 if col not in ("timestamp", "class") else ""
        return df
    except Exception:
        return pd.DataFrame(columns=default_cols)


def render_home_page():
    """Display the home page."""
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-kicker">Circular economy intelligence</div>
            <div class="big-font">AI Waste to Recovery System</div>
            <div class="sub-head">Smart waste sorting and recovery planning powered by TensorFlow, visual AI, and sustainability analytics.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h3>🌍 Goal</h3><p>Reduce landfill waste and improve recovery value.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>🤖 AI</h3><p>Uses MobileNetV2 transfer learning for image classification.</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>📊 Recovery</h3><p>Recommends recycling, composting, reuse, or safe disposal.</p></div>', unsafe_allow_html=True)

    st.markdown("### Project overview")
    st.markdown(
        """
        <div class="premium-card">
            This mini-project captures an image of a waste item, predicts its class, recommends the best recovery path, and estimates the financial and environmental value of each choice.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        - Classify waste into cardboard, glass, metal, paper, plastic, organic, and trash.
        - Estimate the best recovery method and process steps.
        - Evaluate recovery value in INR and CO2 saved per kg.
        - Keep a prediction log for analytics and reporting.
        """
    )


def render_classify_page():
    """Handle waste classification and recommendation."""
    st.markdown("<div class='premium-card'><h3>🧪 Classify Waste</h3></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload a waste image", type=["png", "jpg", "jpeg"])
    camera_photo = st.camera_input("Or capture a photo from webcam")

    source = uploaded_file if uploaded_file is not None else camera_photo
    if source is None:
        st.info("Upload an image or click the camera to classify an item.")
        return

    try:
        image = Image.open(source).convert("RGB")
    except Exception as exc:
        st.error(f"Could not read the image: {exc}")
        return

    st.image(image, caption="Uploaded waste item", use_container_width=True)

    try:
        model, class_names = load_model_and_classes()
    except FileNotFoundError as exc:
        st.error(str(exc))
        return

    source_name = getattr(source, "name", "camera")
    source_size = getattr(source, "size", getattr(source, "_length", "len"))
    cache_key = f"pred_{source_name}_{source_size}"

    if "current_image_key" not in st.session_state or st.session_state.current_image_key != cache_key:
        with st.spinner("Classifying waste item..."):
            image_array = preprocess_image(image)
            predictions = model.predict(image_array, verbose=0)[0]
            predicted_index = int(np.argmax(predictions))
            predicted_class = class_names[predicted_index]
            confidence = float(predictions[predicted_index])
            st.session_state.current_image_key = cache_key
            st.session_state.predicted_class = predicted_class
            st.session_state.confidence = confidence
            st.session_state.predictions = predictions
    else:
        predicted_class = st.session_state.predicted_class
        confidence = st.session_state.confidence
        predictions = st.session_state.predictions

    st.markdown("### Prediction Result")
    st.markdown(f"<div class='premium-card'><h3>✅ Predicted Class: {predicted_class}</h3><p>Confidence: {confidence * 100:.2f}%</p></div>", unsafe_allow_html=True)

    if confidence < 0.60:
        st.warning("⚠️ Low confidence warning: the model is uncertain. Please retry with a clearer image or manually verify the waste category.")

    prob_df = pd.DataFrame({
        "Class": class_names,
        "Confidence": [float(value) for value in predictions]
    }).sort_values("Confidence", ascending=False)
    st.bar_chart(prob_df.set_index("Class"), use_container_width=True)

    weight_kg = st.number_input("Estimated weight (kg)", min_value=0.1, max_value=50.0, value=1.0, step=0.1)
    recovery_info = get_recovery_info(predicted_class, weight_kg)

    st.markdown("### Recovery Recommendation")
    st.markdown(
        f"""
        <div class='premium-card'>
            <h3>♻️ Best Recovery Method: {recovery_info['recovery_method']}</h3>
            <p><strong>Bin color:</strong> {recovery_info['bin_color']}</p>
            <p><strong>Estimated recovery value:</strong> ₹{recovery_info['estimated_value']:.2f}</p>
            <p><strong>CO2 saved:</strong> {recovery_info['estimated_co2_saved']:.2f} kg</p>
            <p><strong>Process steps:</strong></p>
            <ul>
                {''.join(f'<li>{step}</li>' for step in recovery_info['process_steps'])}
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.button("Save prediction"):
        log_prediction(
            class_name=predicted_class,
            confidence=confidence,
            weight_kg=weight_kg,
            value_inr=recovery_info["estimated_value"],
            co2_saved_kg=recovery_info["estimated_co2_saved"],
        )
        st.success("Prediction saved to history.csv")


def render_dashboard_page():
    """Render the dashboard analytics."""
    st.subheader("📊 Recovery Dashboard")
    history_df = summarize_dashboard()

    if history_df.empty:
        st.info("No predictions have been logged yet. Please classify a waste item first.")
        return

    total_items = len(history_df)
    total_value = round(float(pd.to_numeric(history_df["value_inr"], errors="coerce").fillna(0.0).sum()), 2)
    total_co2 = round(float(pd.to_numeric(history_df["co2_saved_kg"], errors="coerce").fillna(0.0).sum()), 2)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total items classified", total_items)
    with col2:
        st.metric("Total estimated value recovered", f"₹{total_value:.2f}")
    with col3:
        st.metric("Total CO2 saved", f"{total_co2:.2f} kg")

    class_count = history_df["class"].value_counts()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("none")
    wedges, texts, autotexts = ax.pie(
        class_count.values,
        labels=class_count.index,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"color": "#edfdf7"},
    )
    for autotext in autotexts:
        autotext.set_color("#0d2c2f")
        autotext.set_weight("bold")
    ax.axis("equal")
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(history_df, use_container_width=True)


def render_about_page():
    """Display project description and usage guidance."""
    st.subheader("ℹ️ About This Project")
    st.markdown(
        """
        This project was developed as a college AIML mini-project to demonstrate how computer vision and machine learning can support sustainable waste management.

        The application uses a MobileNetV2-based computer vision model trained on a waste image dataset, then recommends the most suitable recovery method for the classified material.

        It also estimates financial value and environmental benefit in terms of carbon reduction and recovery efficiency.
        """
    )
    st.markdown("### Tips to improve accuracy")
    st.markdown(
        """
        - Use a clear, well-lit photo of a single waste item.
        - Remove shadows and clutter from the background.
        - Keep the item centered in the frame.
        - Train with diverse images across lighting and angles.
        - Increase the number of labeled examples per class.
        """
    )


def main():
    """Run the Streamlit app."""
    st.set_page_config(page_title="AI Waste to Recovery System", page_icon="♻️", layout="wide")
    load_css()

    navigation = st.sidebar.radio(
        "Navigation",
        ["Home", "Classify Waste", "Dashboard", "About"],
        index=1,
    )

    if navigation == "Home":
        render_home_page()
    elif navigation == "Classify Waste":
        render_classify_page()
    elif navigation == "Dashboard":
        render_dashboard_page()
    else:
        render_about_page()


if __name__ == "__main__":
    main()
