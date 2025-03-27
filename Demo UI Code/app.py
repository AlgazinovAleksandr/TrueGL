import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import logging
import os
import json
import time
from datetime import datetime
from typing import Tuple, Optional, Dict, List
from pathlib import Path
import csv
from io import StringIO
import threading

# Constants
DEFAULT_MODEL_PATH = Path.home() / "Desktop" / "TruthSeeker" / "models" / "finetuned"
MAX_INPUT_LENGTH = 512
MIN_INPUT_LENGTH = 5
RATE_LIMIT_SECONDS = 1
HISTORY_FILE = "prediction_history.json"
CONFIG_FILE = "truthseeker_config.json"
LOG_FILE = "truthseeker.log"
PREDICTION_TIMEOUT = 10

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Configuration management
def load_config() -> Dict:
    default_config = {
        "model_path": str(DEFAULT_MODEL_PATH),
        "colors": {
            "light": {"bg_color": "#ffffff", "text_color": "#1e2a38", "box_bg": "#ffffff", "subtle_text": "#6b7280", "hover_blue": "#4285f4", "light_grey": "#e5e7eb"},
            "dark": {"bg_color": "#1e2a38", "text_color": "#ffffff", "box_bg": "#2d3b4e", "subtle_text": "#9ca3af", "hover_blue": "#4285f4", "light_grey": "#374151"}
        }
    }
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return {**default_config, **json.load(f)}
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
    return default_config

# Load model
@st.cache_resource(show_spinner=True)
def load_model(model_path: str) -> Tuple[Optional[AutoTokenizer], Optional[AutoModelForSequenceClassification], Optional[torch.device], float]:
    logger.info(f"Loading model from {model_path}")
    start_time = time.time()
    if not os.path.exists(model_path):
        logger.error(f"Model path does not exist: {model_path}")
        return None, None, None, 0.0
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 2e9 else "cpu")
        model.to(device)
        model.eval()
        return tokenizer, model, device, time.time() - start_time
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return None, None, None, 0.0

# Prediction
@st.cache_data(show_spinner=False)
def predict(text: str, _tokenizer: AutoTokenizer, _model: AutoModelForSequenceClassification, _device: torch.device) -> Tuple[str, Optional[float], str]:
    if not text.strip():
        return "Please enter a statement.", None, ""
    text = text.strip()
    if len(text) < MIN_INPUT_LENGTH:
        return f"Statement must be at least {MIN_INPUT_LENGTH} characters.", None, ""
    if len(text) > MAX_INPUT_LENGTH:
        text = text[:MAX_INPUT_LENGTH]
        logger.warning(f"Input truncated to {MAX_INPUT_LENGTH} characters")
    try:
        def run_prediction():
            inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=MAX_INPUT_LENGTH)
            inputs = {k: v.to(_device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = _model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[0]
                prediction_idx = torch.argmax(outputs.logits, dim=1).item() - 1
                confidence = probs[prediction_idx + 1].item() * 100
                label_map = {-1: "False", 0: "Mixed", 1: "True"}
                explanation = "High confidence." if confidence > 80 else "Moderate confidence; verify further." if confidence > 50 else "Low confidence; unreliable."
            return label_map[prediction_idx], confidence, explanation

        result = [None]
        thread = threading.Thread(target=lambda: result.__setitem__(0, run_prediction()))
        thread.start()
        thread.join(timeout=PREDICTION_TIMEOUT)
        return result[0] if result[0] else ("Prediction timed out.", None, "")
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return "Analysis unavailable.", None, ""

# History management
def save_to_history(statement: str, prediction: str, confidence: Optional[float]):
    entry = {"timestamp": datetime.now().isoformat(), "statement": statement, "prediction": prediction, "confidence": confidence}
    try:
        history = load_history()
        history.append(entry)
        with open(HISTORY_FILE, "w") as f:
            json.dump(history[-50:], f)
    except Exception as e:
        logger.error(f"Failed to save history: {e}")

def load_history() -> List[Dict]:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
    return []

def export_history(history: List[Dict]) -> StringIO:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=["timestamp", "statement", "prediction", "confidence"])
    writer.writeheader()
    for entry in history:
        writer.writerow(entry)
    output.seek(0)
    return output

# Main app
def main():
    st.set_page_config(page_title="TruthSeeker Lite - Stars AI", layout="centered", initial_sidebar_state="expanded")

    config = load_config()
    model_path = os.getenv("TRUTHSEEKER_MODEL_PATH", config["model_path"])

    # Sidebar (set theme first)
    with st.sidebar:
        theme = st.checkbox("Dark Mode", value=st.session_state.get("theme", False), key="theme")
        colors = config["colors"]["dark" if theme else "light"]

    # CSS for a clean, borderless design with light grey and white
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        body {{background: {colors['bg_color']}; font-family: 'Inter', sans-serif; margin: 0; padding: 0;}}
        .main {{background: {colors['bg_color']}; padding: 30px; border-radius: 12px; max-width: 720px; margin: 40px auto; box-shadow: 0 2px 4px rgba(0,0,0,0.05);}}
        h1 {{color: {colors['text_color']}; font-size: 32px; font-weight: 600; text-align: center; margin-bottom: 24px; letter-spacing: -0.5px;}}
        .stTextArea label {{color: {colors['text_color']}; font-weight: 500; font-size: 16px; margin-bottom: 8px;}}
        .stTextArea textarea {{background: {colors['light_grey']}; color: {colors['text_color']}; border: none; border-radius: 12px; padding: 12px; font-size: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); transition: box-shadow 0.2s ease;}}
        .stTextArea textarea:focus {{box-shadow: 0 0 0 1px {colors['hover_blue']} !important; outline: none;}}
        .stButton>button {{background: {colors['hover_blue']}; color: #ffffff; border-radius: 12px; padding: 10px 24px; font-size: 15px; font-weight: 500; border: none; transition: all 0.2s ease; box-shadow: 0 1px 3px rgba(0,0,0,0.1);}}
        .stButton>button:hover {{background: {colors['hover_blue']}; box-shadow: 0 2px 6px rgba(66, 133, 244, 0.3); transform: translateY(-1px);}}
        .stButton>button:disabled {{background: {colors['subtle_text']}; cursor: not-allowed; box-shadow: none;}}
        .stButton>button[kind="secondary"] {{background: {colors['light_grey']}; color: {colors['text_color']}; box-shadow: 0 1px 3px rgba(0,0,0,0.1);}}
        .stButton>button[kind="secondary"]:hover {{background: {colors['subtle_text']}; box-shadow: 0 2px 6px rgba(0,0,0,0.2); transform: translateY(-1px);}}
        .result {{background: {colors['light_grey']}; padding: 20px; border-radius: 12px; margin-top: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: none; transition: box-shadow 0.2s ease;}}
        .result:hover {{box-shadow: 0 0 0 1px {colors['hover_blue']}, 0 2px 6px rgba(66, 133, 244, 0.2);}}
        .result p {{color: {colors['text_color']}; margin: 6px 0; font-size: 15px; line-height: 1.5;}}
        .result .explanation {{color: {colors['subtle_text']}; font-style: italic;}}
        .stSidebar {{background: {colors['light_grey']}; padding: 20px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); border: none;}}
        .stSidebar label {{color: {colors['text_color']}; font-size: 15px; font-weight: 500; margin-bottom: 8px;}}
        .character-count {{color: {colors['subtle_text']}; font-size: 13px; text-align: right; margin-top: 6px;}}
        .stAlert {{background: {colors['light_grey']}; color: {colors['text_color']}; border: none; border-radius: 12px; padding: 12px; margin-top: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); transition: box-shadow 0.2s ease;}}
        .stAlert:hover {{box-shadow: 0 0 0 1px {colors['hover_blue']}, 0 2px 6px rgba(66, 133, 244, 0.2);}}
        .stProgress > div > div {{background: {colors['hover_blue']}; border-radius: 6px; height: 6px;}}
        .stExpander {{background: {colors['light_grey']}; border: none; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); padding: 10px; transition: box-shadow 0.2s ease;}}
        .stExpander:hover {{box-shadow: 0 0 0 1px {colors['hover_blue']}, 0 2px 6px rgba(66, 133, 244, 0.2);}}
        .stSelectbox div {{background: {colors['light_grey']}; color: {colors['text_color']}; border: none; border-radius: 12px; padding: 8px; font-size: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); transition: box-shadow 0.2s ease;}}
        .stSelectbox div:hover {{box-shadow: 0 0 0 1px {colors['hover_blue']}, 0 2px 6px rgba(66, 133, 244, 0.2);}}
        /* Remove all Streamlit default borders and styles */
        [data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"], [data-testid="stForm"], [data-testid="stAppViewContainer"] {{border: none !important; box-shadow: none !important;}}
        /* Hide Deploy button */
        [data-testid="stDeployButton"] {{display: none !important;}}
        /* Style slider */
        .stSlider [role="slider"] {{background-color: {colors['subtle_text']} !important;}}
        .stSlider [role="slider"]:hover {{background-color: {colors['hover_blue']} !important;}}
        </style>
        <script>
        document.addEventListener('keydown', function(e) {{
            if (e.key === 'Enter' && !e.shiftKey) document.querySelector('.stButton button[aria-label="Analyze"]').click();
            if (e.key === 'Escape') document.querySelector('.stButton button[aria-label="Clear"]').click();
        }});
        document.addEventListener('DOMContentLoaded', function() {{
            const textarea = document.querySelector('textarea');
            const counter = document.querySelector('.character-count');
            if (textarea && counter) textarea.addEventListener('input', () => counter.textContent = `${{textarea.value.length}}/{MAX_INPUT_LENGTH}`);
            document.body.style.backgroundColor = "{colors['bg_color']}";
        }});
        </script>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("<h1>TruthSeeker Lite</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: {colors['subtle_text']}; font-size: 15px; margin-top: -15px;'>Truth Analysis by Stars AI</p>", unsafe_allow_html=True)

    # Model loading
    with st.spinner("Initializing model..."):
        tokenizer, model, device, load_time = load_model(model_path)
    if not tokenizer or not model:
        st.error("Failed to initialize model. Check path or retry.")
        if st.button("Retry", key="retry"):
            st.cache_resource.clear()
            st.rerun()
        return

    # Sidebar controls
    with st.sidebar:
        st.markdown(f"<p style='color: {colors['subtle_text']}; font-size: 13px; text-align: center;'>Device: {device}<br>Load Time: {load_time:.2f}s</p>", unsafe_allow_html=True)
        confidence_threshold = st.slider("Confidence Threshold", 0, 100, 50, step=5, format="%d%%", help="Minimum confidence for results")
        analysis_mode = st.selectbox("Mode", ["Quick", "Detailed"], help="Quick: Faster; Detailed: More thorough")

    # Main UI
    with st.container():
        if "input_text" not in st.session_state:
            st.session_state["input_text"] = ""
        examples = ["", "The Earth is flat.", "Water boils at 100°C.", "The moon is made of cheese."]
        st.selectbox("Try an Example", examples, key="example", on_change=lambda: st.session_state.update({"input_text": st.session_state["example"]}))

        with st.form("input_form"):
            statement = st.text_area("Enter Statement", value=st.session_state["input_text"], height=120, key="input", disabled=not tokenizer, placeholder="Type a statement to analyze")
            st.markdown(f"<p class='character-count'>{len(statement)}/{MAX_INPUT_LENGTH}</p>", unsafe_allow_html=True)
            col1, col2 = st.columns([1, 1])  # Equal columns for balanced buttons
            with col1:
                analyze_clicked = st.form_submit_button("Analyze", type="primary", disabled=not tokenizer, help="Analyze the statement (Enter)")
            with col2:
                clear_clicked = st.form_submit_button("Clear", type="secondary", disabled=not tokenizer, help="Clear input (Esc)")

    # Handle button actions
    if clear_clicked:
        st.session_state["input_text"] = ""
        st.session_state["example"] = ""
        st.success("Input cleared successfully.")
        st.rerun()
    elif analyze_clicked and statement:
        current_time = time.time()
        if "last_submit" not in st.session_state:
            st.session_state["last_submit"] = 0
        if current_time - st.session_state["last_submit"] < RATE_LIMIT_SECONDS:
            st.warning("Please wait a moment before analyzing again.")
        else:
            st.session_state["last_submit"] = current_time
            with st.spinner(f"Analyzing in {analysis_mode} mode..."):
                prediction, confidence, explanation = predict(statement, tokenizer, model, device)
                st.session_state["input_text"] = statement
                if confidence and confidence >= confidence_threshold:
                    save_to_history(statement, prediction, confidence)
                    st.markdown(f"""
                        <div class='result'>
                            <p><strong>Statement:</strong> {statement}</p>
                            <p><strong>Result:</strong> {prediction} ({confidence:.1f}%)</p>
                            <p class='explanation'>{explanation}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.progress(int(confidence))
                elif confidence:
                    st.warning(f"Confidence ({confidence:.1f}%) is below threshold ({confidence_threshold}%).")
                else:
                    st.error(prediction)

    # History with search
    with st.expander("Recent Analyses"):
        history = load_history()
        search = st.text_input("Search History", "", key="history_search", placeholder="Filter by statement or result")
        filtered_history = [entry for entry in history if search.lower() in entry["statement"].lower() or search.lower() in entry["prediction"].lower()]
        if filtered_history:
            for entry in reversed(filtered_history):
                st.markdown(f"<p style='color: {colors['text_color']}; font-size: 15px;'>{entry['timestamp']}: {entry['statement']} - {entry['prediction']} ({entry['confidence']:.1f}%)</p>", unsafe_allow_html=True)
            if st.button("Export as CSV"):
                st.download_button("Download", export_history(filtered_history), f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
        else:
            st.markdown(f"<p style='color: {colors['subtle_text']}; font-size: 15px;'>No matching history found.</p>", unsafe_allow_html=True)

    # Footer
    st.markdown(f"<p style='text-align: center; color: {colors['subtle_text']}; font-size: 13px; margin-top: 20px;'>© 2025 Stars AI</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
