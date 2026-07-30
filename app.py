# ============================================================
# SARCASM DETECTION - 4 MODEL COMPARISON DASHBOARD
# ============================================================

import os
import time
import pickle
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import torch
import joblib
import gdown

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
from peft import PeftModel


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Sarcasm Detection | Model Comparison",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

warnings.filterwarnings("ignore")


# ============================================================
# 2. PATHS / CONSTANTS + STREAMLIT CLOUD MODEL DOWNLOAD
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GOOGLE_DRIVE_FOLDER_URL = (
    "https://drive.google.com/drive/folders/"
    "1I_p5tWHQpia4jcSLm7XexfF8yrhgpNM8?usp=drive_link"
)

# Models downloaded from Google Drive are stored here at runtime.
MODEL_DIR = os.path.join(BASE_DIR, "sarcastic_models")

ML_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "tuned_tfidf_linear_svm.pkl"
)

DL_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "tuned_bilstm.keras"
)

DL_TOKENIZER_PATH = os.path.join(
    MODEL_DIR,
    "tokenizer_dl.pkl"
)

LORA_BASE_MODEL = "distilbert-base-uncased"
PRETRAINED_MODEL_NAME = "helinivan/english-sarcasm-detector"

MAX_LEN_DL = 16
TRANSFORMER_MAX_LEN = 128

LABEL_MAP = {
    0: "Not Sarcastic",
    1: "Sarcastic"
}


def find_lora_adapter_dir():
    """
    Find the downloaded LoRA directory by locating adapter_config.json.
    This avoids depending on the exact Google Drive LoRA folder name.
    """
    if not os.path.isdir(MODEL_DIR):
        return None

    for root, _, files in os.walk(MODEL_DIR):
        if (
            "adapter_config.json" in files
            and "adapter_model.safetensors" in files
        ):
            return root

    return None


def core_drive_files_exist():
    """Check whether the ML and DL artifacts are already available."""
    return all(
        os.path.isfile(path)
        for path in [
            ML_MODEL_PATH,
            DL_MODEL_PATH,
            DL_TOKENIZER_PATH,
        ]
    )


def all_drive_models_exist():
    """Check whether all Google Drive model artifacts are available."""
    return (
        core_drive_files_exist()
        and find_lora_adapter_dir() is not None
    )


@st.cache_resource(show_spinner=False)
def download_model_files():
    """
    Download Models 1-3 from the shared Google Drive folder.
    Model 4 is downloaded separately from Hugging Face.
    """

    if all_drive_models_exist():
        return MODEL_DIR

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    gdown.download_folder(
        url=GOOGLE_DRIVE_FOLDER_URL,
        output=MODEL_DIR,
        quiet=False,
        use_cookies=False
    )

    if not all_drive_models_exist():
        raise FileNotFoundError(
            "The Google Drive folder was downloaded, but one or more "
            "required model artifacts could not be found. Confirm that "
            "the folder is shared as 'Anyone with the link - Viewer' "
            "and contains tuned_tfidf_linear_svm.pkl, tuned_bilstm.keras, "
            "tokenizer_dl.pkl, adapter_config.json, and "
            "adapter_model.safetensors."
        )

    return MODEL_DIR


# ============================================================
# 3. CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       MAIN APP
    ======================================================== */

    .stApp {
        background-color: #f7f8fc;
    }


    /* ========================================================
       SIDEBAR
    ======================================================== */

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #071426 0%,
            #0c1c33 100%
        );
    }

    [data-testid="stSidebar"] * {
        color: white;
    }


    /* ========================================================
       MAIN TITLE
    ======================================================== */

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 2px;
        text-align: center;
    }

    .subtitle {
        color: #64748b;
        font-size: 15px;
        margin-bottom: 18px;
        text-align: center;
    }


    /* ========================================================
       METRIC CARDS
    ======================================================== */

    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;

        padding: 20px;

        min-height: 135px;

        box-shadow:
            0 2px 8px rgba(0, 0, 0, 0.04);
    }

    .metric-label {
        color: #64748b;
        font-size: 14px;
        font-weight: 600;
    }

    .metric-value {
        color: #111827;
        font-size: 31px;
        font-weight: 800;
        margin-top: 6px;
    }

    .metric-caption {
        color: #64748b;
        font-size: 13px;
        margin-top: 4px;
    }


    /* ========================================================
       GENERAL SECTION CARD
    ======================================================== */

    .section-card {
        background: white;

        border: 1px solid #e5e7eb;
        border-radius: 16px;

        padding: 22px;

        box-shadow:
            0 2px 8px rgba(0, 0, 0, 0.04);

        margin-bottom: 15px;
    }


    /* ========================================================
       PREDICTION CARDS
    ======================================================== */

    .pred-card {

        background: linear-gradient(
            145deg,
            #ffffff 0%,
            #f8fafc 100%
        );

        border: 1px solid #dbe1ea;
        border-radius: 18px;

        padding: 18px 12px;

        height: 285px;

        box-sizing: border-box;

        text-align: center;

        box-shadow:
            0 8px 22px rgba(15, 23, 42, 0.07);

        transition:
            transform 0.25s ease,
            box-shadow 0.25s ease,
            border-color 0.25s ease;
    }


    /* Prediction card hover */

    .pred-card:hover {

        transform: translateY(-4px);

        border-color: #c7d2fe;

        box-shadow:
            0 14px 30px rgba(79, 70, 229, 0.12);
    }


    /* ========================================================
       PREDICTION ICON
    ======================================================== */

    .pred-icon {

        height: 55px;

        display: flex;
        align-items: center;
        justify-content: center;

        font-size: 32px;

        line-height: 1;

        margin-bottom: 4px;
    }


    /* ========================================================
       MODEL CATEGORY
       MACHINE LEARNING / DEEP LEARNING / ETC.
    ======================================================== */

    .pred-category {

        height: 48px;

        display: flex;
        align-items: center;
        justify-content: center;

        text-align: center;

        font-size: 10px;
        font-weight: 800;

        letter-spacing: 0.8px;

        line-height: 1.5;

        color: #64748b;

        text-transform: uppercase;

        padding: 0 3px;

        box-sizing: border-box;
    }


    /* ========================================================
       MODEL NAME
    ======================================================== */

    .pred-model {

        height: 68px;

        display: flex;
        align-items: center;
        justify-content: center;

        text-align: center;

        font-size: 14px;
        font-weight: 750;

        line-height: 1.5;

        color: #0f172a;

        padding: 0 4px;

        box-sizing: border-box;
    }


    /* ========================================================
       PREDICTION RESULT COMMON STYLE
    ======================================================== */

    .pred-result {

        height: 65px;

        display: flex;
        align-items: center;
        justify-content: center;

        text-align: center;

        border-radius: 12px;

        padding: 6px;

        box-sizing: border-box;

        font-size: 13px;
        font-weight: 800;

        letter-spacing: 0.3px;

        line-height: 1.5;
    }


    /* ========================================================
       SARCASTIC RESULT
    ======================================================== */

    .pred-sarcastic {

        background: linear-gradient(
            135deg,
            #ecfdf5,
            #dcfce7
        );

        color: #15803d;

        border: 1px solid #86efac;

        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.8),
            0 3px 8px rgba(22, 163, 74, 0.08);
    }


    /* ========================================================
       NOT SARCASTIC RESULT
    ======================================================== */

    .pred-not-sarcastic {

        background: linear-gradient(
            135deg,
            #fffbea,
            #fef3c7
        );

        color: #a16207;

        border: 1px solid #fde68a;

        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.8),
            0 3px 8px rgba(234, 179, 8, 0.08);
    }


    /* ========================================================
       STREAMLIT BORDERED CONTAINERS
       USED BY THE CURRENT PREDICTION CARDS
    ======================================================== */

    [data-testid="stHorizontalBlock"]
    [data-testid="stVerticalBlockBorderWrapper"] {

        min-height: 330px;

        height: 330px;

        box-sizing: border-box;
    }


    [data-testid="stHorizontalBlock"]
    [data-testid="stVerticalBlockBorderWrapper"] > div {

        height: 100%;

        box-sizing: border-box;
    }


    [data-testid="stHorizontalBlock"]
    [data-testid="stVerticalBlockBorderWrapper"]
    [data-testid="stVerticalBlock"] {

        height: 100%;
    }


    /* ========================================================
       MODEL AGREEMENT BOX
    ======================================================== */

    .agreement-box {

        background: linear-gradient(
            135deg,
            #eef2ff,
            #f5f3ff
        );

        border: 1px solid #c7d2fe;

        padding: 16px 18px;

        border-radius: 12px;

        margin-top: 16px;

        color: #312e81;

        box-shadow:
            0 5px 18px rgba(79, 70, 229, 0.07);
    }


    /* ========================================================
       SMALL MUTED TEXT
    ======================================================== */

    .small-muted {

        color: #64748b;

        font-size: 13px;
    }


    /* ========================================================
       STREAMLIT BUTTON
    ======================================================== */

    div.stButton > button {

        background: linear-gradient(
            90deg,
            #4f46e5,
            #7c3aed
        );

        color: white;

        border: none;

        border-radius: 9px;

        font-weight: 700;

        padding: 0.65rem 1.2rem;

        box-shadow:
            0 5px 14px rgba(79, 70, 229, 0.18);

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }


    /* Button hover */

    div.stButton > button:hover {

        color: white;

        border: none;

        transform: translateY(-1px);

        box-shadow:
            0 8px 18px rgba(79, 70, 229, 0.25);
    }


    /* ========================================================
       DATAFRAME
    ======================================================== */

    [data-testid="stDataFrame"] {

        border-radius: 12px;

        overflow: hidden;
    }


    /* ========================================================
       DIVIDER
    ======================================================== */

    hr {

        border: none;

        border-top: 1px solid #e2e8f0;

        margin-top: 25px;

        margin-bottom: 25px;
    }


    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 4. MODEL PERFORMANCE DATA
# ============================================================

performance_data = pd.DataFrame(
    {
        "Model": [
            "TF-IDF + Linear SVM",
            "Tuned Bi-LSTM",
            "DistilBERT + LoRA",
            "Pretrained Transformer",
        ],

        "Train Accuracy": [
            0.953995,
            0.910710,
            np.nan,
            np.nan,
        ],

        "Test Accuracy": [
            0.793896,
            0.861779,
            0.8593,
            0.9400,
        ],

        "Precision": [
            np.nan,
            np.nan,
            0.8561,
            0.9618,
        ],

        "Recall": [
            0.772778,
            0.871265,
            0.8466,
            0.9100,
        ],

        "F1 Score": [
            np.nan,
            np.nan,
            0.8513,
            0.9352,
        ],
    }
)


# ============================================================
# 5. LOAD ML MODEL
# ============================================================

@st.cache_resource
def load_ml_model():

    model = joblib.load(ML_MODEL_PATH)

    return model


# ============================================================
# 6. LOAD BI-LSTM
# ============================================================

@st.cache_resource
def load_dl_model():

    model = load_model(
        DL_MODEL_PATH,
        compile=False
    )

    with open(DL_TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)

    return model, tokenizer


# ============================================================
# 7. LOAD LoRA MODEL
# ============================================================

@st.cache_resource
def load_lora_model():

    tokenizer = AutoTokenizer.from_pretrained(
        find_lora_adapter_dir()
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        LORA_BASE_MODEL,
        num_labels=2
    )

    model = PeftModel.from_pretrained(
        base_model,
        find_lora_adapter_dir()
    )

    model.eval()

    return tokenizer, model


# ============================================================
# 8. LOAD HELINIVAN PRETRAINED MODEL
# ============================================================

@st.cache_resource
def load_pretrained_model():

    tokenizer = AutoTokenizer.from_pretrained(
        PRETRAINED_MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME
    )

    model.eval()

    return tokenizer, model


# ============================================================
# 9. LOAD EVERYTHING
# ============================================================

@st.cache_resource
def load_all_models():

    download_model_files()

    ml_model = load_ml_model()

    dl_model, dl_tokenizer = load_dl_model()

    lora_tokenizer, lora_model = load_lora_model()

    pretrained_tokenizer, pretrained_model = (
        load_pretrained_model()
    )

    return (
        ml_model,
        dl_model,
        dl_tokenizer,
        lora_tokenizer,
        lora_model,
        pretrained_tokenizer,
        pretrained_model,
    )


# ============================================================
# 10. ML PREDICTION
# ============================================================

def predict_ml(text, model):

    start = time.perf_counter()

    prediction = int(model.predict([text])[0])

    # LinearSVC gives decision_function, not calibrated probability.
    decision_score = None

    if hasattr(model, "decision_function"):

        score = model.decision_function([text])

        score = np.asarray(score)

        if score.ndim == 2:
            score = score[0]

        if score.size == 1:
            decision_score = float(score.reshape(-1)[0])

        else:
            decision_score = float(np.max(score))

    elapsed = (
        time.perf_counter() - start
    ) * 1000

    return {
        "prediction": LABEL_MAP[prediction],
        "class_id": prediction,
        "confidence": None,
        "sarcastic_probability": None,
        "decision_score": decision_score,
        "time_ms": elapsed,
    }


# ============================================================
# 11. BI-LSTM PREDICTION
# ============================================================

def predict_dl(text, model, tokenizer):

    start = time.perf_counter()

    sequence = tokenizer.texts_to_sequences(
        [text]
    )

    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN_DL,
        padding="post",
        truncating="post"
    )

    probability = float(
        model.predict(
            padded,
            verbose=0
        )[0][0]
    )

    prediction = (
        1 if probability >= 0.5 else 0
    )

    confidence = (
        probability
        if prediction == 1
        else 1 - probability
    )

    elapsed = (
        time.perf_counter() - start
    ) * 1000

    return {
        "prediction": LABEL_MAP[prediction],
        "class_id": prediction,
        "confidence": confidence,
        "sarcastic_probability": probability,
        "decision_score": None,
        "time_ms": elapsed,
    }


# ============================================================
# 12. GENERIC TRANSFORMER PREDICTION
# ============================================================

def transformer_raw_prediction(
    text,
    tokenizer,
    model
):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=TRANSFORMER_MAX_LEN
    )

    # DistilBERT does NOT use token_type_ids.
    # Some saved tokenizers can still return this field,
    # so remove it before passing inputs to the model.
    inputs.pop("token_type_ids", None)

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    raw_class = int(
        torch.argmax(
            probabilities
        ).item()
    )

    raw_confidence = float(
        probabilities[
            raw_class
        ].item()
    )

    raw_probability_class_1 = float(
        probabilities[1].item()
    )

    return (
        raw_class,
        raw_confidence,
        raw_probability_class_1
    )


# ============================================================
# 13. LoRA PREDICTION
# ============================================================

def predict_lora(
    text,
    tokenizer,
    model
):

    start = time.perf_counter()

    (
        prediction,
        confidence,
        probabilities
    ) = transformer_raw_prediction(
        text,
        tokenizer,
        model
    )

    sarcastic_probability = float(
        probabilities[1]
    )

    elapsed = (
        time.perf_counter() - start
    ) * 1000

    return {
        "prediction": LABEL_MAP[prediction],
        "class_id": prediction,
        "confidence": confidence,
        "sarcastic_probability": sarcastic_probability,
        "decision_score": None,
        "time_ms": elapsed,
    }


# ============================================================
# 14. PRETRAINED MODEL LABEL NORMALIZATION
# ============================================================

def infer_sarcastic_class_id(model):

    id2label = getattr(
        model.config,
        "id2label",
        {}
    )

    if not id2label:
        return 1

    for class_id, label in id2label.items():

        label_text = str(label).lower()

        if (
            "sarcas" in label_text
            and "not" not in label_text
            and "non" not in label_text
        ):
            return int(class_id)

    # Fallback based on our verified project mapping
    return 1


# ============================================================
# 15. PRETRAINED MODEL PREDICTION
# ============================================================

def predict_pretrained(
    text,
    tokenizer,
    model
):

    start = time.perf_counter()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=TRANSFORMER_MAX_LEN
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    raw_predicted_id = int(
        torch.argmax(probabilities).item()
    )

    sarcastic_class_id = (
        infer_sarcastic_class_id(model)
    )

    prediction = (
        1
        if raw_predicted_id == sarcastic_class_id
        else 0
    )

    confidence = float(
        probabilities[raw_predicted_id].item()
    )

    sarcastic_probability = float(
        probabilities[sarcastic_class_id].item()
    )

    elapsed = (
        time.perf_counter() - start
    ) * 1000

    return {
        "prediction": LABEL_MAP[prediction],
        "class_id": prediction,
        "confidence": confidence,
        "sarcastic_probability": sarcastic_probability,
        "decision_score": None,
        "time_ms": elapsed,
    }


# ============================================================
# 16. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        ## 🧠 SARCASM DETECTION

        **NLP Model Comparison**
        """
    )

    st.markdown("---")

    st.markdown("### 🏠 Home")

    st.markdown("### 🔍 Prediction")
    st.markdown("Predict Sarcasm")

    st.markdown("### 📊 Model Comparison")
    st.markdown("Performance Dashboard")
    st.markdown("Detailed Comparison")

    st.markdown("### 📈 Analytics")
    st.markdown("Model Metrics")

    st.markdown("### ℹ️ About")
    st.markdown("About Models")
    st.markdown("Dataset Info")

    st.markdown("---")

    st.markdown(
        """
        Built with ❤️ using

        **Streamlit**

        **TensorFlow**

        **PyTorch**

        **Hugging Face**
        """
    )






# ============================================================
# 17. HEADER
# ============================================================

st.markdown(
    """
    <h1 style="
        text-align: center;
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 8px;
    ">
        Sarcasm Detection — Model Comparison Dashboard ✨
    </h1>

    <p style="
        text-align: center;
        font-size: 16px;
        color: #64748b;
        margin-top: 0px;
        margin-bottom: 25px;
    ">
        Compare Machine Learning, Deep Learning and Transformer-based
        approaches for sarcasm detection.
    </p>
    """,
    unsafe_allow_html=True
)



# ============================================================
# 18. MODEL OVERVIEW / TOP METRIC CARDS
# ============================================================

average_accuracy = (
    performance_data["Test Accuracy"].mean() * 100
)

st.markdown(
    """
    <div style="text-align:center; margin-top:10px; margin-bottom:5px;">
        <div style="
            font-size:23px;
            font-weight:800;
            color:#26355d;
        ">
            ✨ Sarcasm Detection Model Overview ✨
        </div>
        <div style="
            font-size:12px;
            color:#7c879e;
            margin-top:5px;
        ">
            Performance Summary Across All Models
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="
        height:2px;
        margin:12px auto 20px auto;
        width:65%;
        background:linear-gradient(
            90deg,
            transparent,
            #8b5cf6,
            #60a5fa,
            transparent
        );
        border-radius:10px;
    "></div>
    """,
    unsafe_allow_html=True
)


# ------------------------------------------------------------
# FOUR OVERVIEW CARDS
# ------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4, gap="medium")


# ============================================================
# CARD 1 - DATASET SIZE
# ============================================================

with c1:

    st.markdown(
        """
        <div style="background:linear-gradient(145deg,#ffffff,#faf7ff);border:1px solid #ede9fe;border-radius:18px;padding:20px 15px;text-align:center;min-height:190px;box-shadow:0 7px 22px rgba(124,58,237,0.10);">
            <div style="width:46px;height:46px;margin:-5px auto 12px auto;border-radius:50%;background:#f3e8ff;display:flex;align-items:center;justify-content:center;font-size:23px;box-shadow:0 5px 14px rgba(124,58,237,0.15);">🗄️</div>
            <div style="font-size:12px;font-weight:700;color:#475569;">Dataset Size</div>
            <div style="font-size:30px;font-weight:850;color:#7c3aed;margin:10px 0 8px 0;">25,713</div>
            <div style="width:35px;height:2px;background:#8b5cf6;margin:0 auto 12px auto;border-radius:5px;"></div>
            <div style="font-size:11px;color:#94a3b8;">Headlines</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CARD 2 - TOTAL MODELS
# ============================================================

with c2:

    st.markdown(
        """
        <div style="background:linear-gradient(145deg,#ffffff,#f5f9ff);border:1px solid #dbeafe;border-radius:18px;padding:20px 15px;text-align:center;min-height:190px;box-shadow:0 7px 22px rgba(37,99,235,0.09);">
            <div style="width:46px;height:46px;margin:-5px auto 12px auto;border-radius:50%;background:#e0f2fe;display:flex;align-items:center;justify-content:center;font-size:23px;box-shadow:0 5px 14px rgba(37,99,235,0.13);">🧩</div>
            <div style="font-size:12px;font-weight:700;color:#475569;">Total Models</div>
            <div style="font-size:30px;font-weight:850;color:#2563eb;margin:10px 0 8px 0;">4</div>
            <div style="width:35px;height:2px;background:#3b82f6;margin:0 auto 12px auto;border-radius:5px;"></div>
            <div style="font-size:11px;color:#94a3b8;">Models Compared</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CARD 3 - BEST F1 SCORE
# ============================================================

with c3:

    st.markdown(
        """
        <div style="position:relative;background:linear-gradient(145deg,#ffffff,#f0fdf4);border:1px solid #86efac;border-radius:18px;padding:20px 15px;text-align:center;min-height:190px;box-shadow:0 8px 26px rgba(22,163,74,0.13);">
            <div style="position:absolute;right:10px;top:10px;background:#16a34a;color:white;font-size:9px;font-weight:800;padding:4px 8px;border-radius:20px;">★ BEST</div>
            <div style="width:46px;height:46px;margin:-5px auto 12px auto;border-radius:50%;background:#dcfce7;display:flex;align-items:center;justify-content:center;font-size:23px;box-shadow:0 5px 14px rgba(22,163,74,0.16);">🎯</div>
            <div style="font-size:12px;font-weight:700;color:#475569;">Best F1 Score</div>
            <div style="font-size:30px;font-weight:850;color:#16a34a;margin:10px 0 8px 0;">93.52%</div>
            <div style="width:35px;height:2px;background:#22c55e;margin:0 auto 12px auto;border-radius:5px;"></div>
            <div style="font-size:11px;color:#94a3b8;">Pretrained Transformer</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# CARD 4 - AVERAGE ACCURACY
# ============================================================

with c4:

    st.markdown(
        f"""
        <div style="background:linear-gradient(145deg,#ffffff,#fff7ed);border:1px solid #fed7aa;border-radius:18px;padding:20px 15px;text-align:center;min-height:190px;box-shadow:0 7px 22px rgba(234,88,12,0.10);">
            <div style="width:46px;height:46px;margin:-5px auto 12px auto;border-radius:50%;background:#ffedd5;display:flex;align-items:center;justify-content:center;font-size:23px;box-shadow:0 5px 14px rgba(234,88,12,0.14);">📊</div>
            <div style="font-size:12px;font-weight:700;color:#475569;">Average Test Accuracy</div>
            <div style="font-size:30px;font-weight:850;color:#ea580c;margin:10px 0 8px 0;">{average_accuracy:.2f}%</div>
            <div style="width:35px;height:2px;background:#f97316;margin:0 auto 12px auto;border-radius:5px;"></div>
            <div style="font-size:11px;color:#94a3b8;">Across All Models</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")


# ============================================================
# 19. LOAD MODELS
# ============================================================

try:

    with st.spinner(
        "Downloading model files if needed and loading all four models..."
    ):

        (
            ml_model,
            dl_model,
            dl_tokenizer,
            lora_tokenizer,
            lora_model,
            pretrained_tokenizer,
            pretrained_model,
        ) = load_all_models()

    models_loaded = True

except Exception as error:

    models_loaded = False

    st.error(
        "One or more models could not be loaded."
    )

    st.exception(error)


# ============================================================
# 20. INPUT + RESULTS AREA
# ============================================================

left, right = st.columns(
    [0.92, 1.35],
    gap="large"
)


# ---------------- INPUT ----------------

with left:

    st.markdown("### 📝 Test a Headline")

    st.caption(
        "Enter a headline or sentence and compare "
        "predictions from all four models."
    )

    headline = st.text_area(
        "Headline",
        value="Oh great, another Monday morning!",
        height=145,
        label_visibility="collapsed"
    )

    predict_button = st.button(
        "🚀 Predict with All Models",
        use_container_width=True,
        disabled=not models_loaded
    )

    st.caption(
        "Try literal and sarcastic examples to see "
        "where the models agree or disagree."
    )


# ============================================================
# 21. RUN PREDICTIONS
# ============================================================

if predict_button:

    if not headline.strip():

        st.warning(
            "Please enter a headline first."
        )

    else:

        with st.spinner(
            "Running all four models..."
        ):

            ml_result = predict_ml(
                headline,
                ml_model
            )

            dl_result = predict_dl(
                headline,
                dl_model,
                dl_tokenizer
            )

            lora_result = predict_lora(
                headline,
                lora_tokenizer,
                lora_model
            )

            pretrained_result = (
                predict_pretrained(
                    headline,
                    pretrained_tokenizer,
                    pretrained_model
                )
            )

        st.session_state["results"] = {
            "TF-IDF + Linear SVM":
                ml_result,

            "Tuned Bi-LSTM":
                dl_result,

            "DistilBERT + LoRA":
                lora_result,

            "Pretrained Transformer":
                pretrained_result,
        }


# ============================================================
# 22. DISPLAY PREDICTION RESULTS
# ============================================================

with right:

    st.markdown("### 📌 Prediction Results")

    if "results" not in st.session_state:

        st.info(
            "Enter a headline and click "
            "'Predict with All Models' to compare predictions."
        )

    else:

        results = st.session_state["results"]

        # ====================================================
        # MODEL INFORMATION
        # ====================================================

        model_info = {

            "TF-IDF + Linear SVM": {
                "icon": "⚙️",
                "type": "MACHINE LEARNING"
            },

            "Tuned Bi-LSTM": {
                "icon": "🧠",
                "type": "DEEP LEARNING"
            },

            "DistilBERT + LoRA": {
                "icon": "⚡",
                "type": "PEFT TRANSFORMER"
            },

            "Pretrained Transformer": {
                "icon": "🤗",
                "type": "PRETRAINED TRANSFORMER"
            }
        }


        # ====================================================
        # FOUR MODEL COLUMNS
        # ====================================================

        cols = st.columns(
            4,
            gap="small"
        )


        for col, (model_name, result) in zip(
            cols,
            results.items()
        ):

            info = model_info.get(
                model_name,
                {
                    "icon": "🤖",
                    "type": "MODEL"
                }
            )


            # =================================================
            # PREDICTION STYLE
            # =================================================

            if result["prediction"] == "Sarcastic":

                prediction_emoji = "😏"
                prediction_text = "SARCASTIC"

                prediction_bg = "#ecfdf5"
                prediction_border = "#86efac"
                prediction_color = "#15803d"

            else:

                prediction_emoji = "🙂"
                prediction_text = "NOT SARCASTIC"

                prediction_bg = "#fffbea"
                prediction_border = "#fde68a"
                prediction_color = "#a16207"


            # =================================================
            # COMPLETE FIXED-SIZE CARD
            # =================================================
            #
            # Important:
            # The HTML is intentionally kept compact.
            # This prevents Streamlit from interpreting
            # indented HTML as Markdown code.
            # =================================================

            card_html = (
                '<div style="'
                'height:300px;'
                'box-sizing:border-box;'
                'display:flex;'
                'flex-direction:column;'
                'align-items:center;'
                'justify-content:flex-start;'
                'text-align:center;'
                'padding:10px 4px;'
                '">'

                # ICON
                '<div style="'
                'height:55px;'
                'width:100%;'
                'display:flex;'
                'align-items:center;'
                'justify-content:center;'
                'font-size:32px;'
                '">'
                f'{info["icon"]}'
                '</div>'

                # MODEL TYPE
                '<div style="'
                'height:55px;'
                'width:100%;'
                'display:flex;'
                'align-items:center;'
                'justify-content:center;'
                'font-size:11px;'
                'font-weight:800;'
                'letter-spacing:0.6px;'
                'color:#64748b;'
                'line-height:1.5;'
                'padding:0 4px;'
                'box-sizing:border-box;'
                '">'
                f'{info["type"]}'
                '</div>'

                # MODEL NAME
                '<div style="'
                'height:90px;'
                'width:100%;'
                'display:flex;'
                'align-items:center;'
                'justify-content:center;'
                'font-size:16px;'
                'font-weight:750;'
                'color:#0f172a;'
                'line-height:1.5;'
                'padding:0 5px;'
                'box-sizing:border-box;'
                '">'
                f'{model_name}'
                '</div>'

                # PREDICTION RESULT
                '<div style="'
                'height:72px;'
                'width:100%;'
                'display:flex;'
                'align-items:center;'
                'justify-content:center;'
                f'background:{prediction_bg};'
                f'border:1px solid {prediction_border};'
                'border-radius:12px;'
                f'color:{prediction_color};'
                'font-size:14px;'
                'font-weight:800;'
                'line-height:1.5;'
                'padding:8px;'
                'box-sizing:border-box;'
                '">'
                f'{prediction_emoji}&nbsp;&nbsp;'
                f'{prediction_text}'
                '</div>'

                '</div>'
            )


            # =================================================
            # DISPLAY CARD
            # =================================================

            with col:

                with st.container(
                    border=True
                ):

                    st.markdown(
                        card_html,
                        unsafe_allow_html=True
                    )


        # ====================================================
        # MODEL AGREEMENT
        # ====================================================

        predictions = [
            result["class_id"]
            for result in results.values()
        ]

        sarcastic_votes = predictions.count(1)

        not_sarcastic_votes = predictions.count(0)

        total_models = len(predictions)


        # Space between cards and agreement box

        st.write("")


        # ====================================================
        # AGREEMENT RESULT
        # ====================================================

        if sarcastic_votes == total_models:

            st.success(
                f"🤝 All {total_models} models agree: "
                "😏 Sarcastic"
            )


        elif not_sarcastic_votes == total_models:

            st.info(
                f"🤝 All {total_models} models agree: "
                "🙂 Not Sarcastic"
            )


        elif sarcastic_votes > not_sarcastic_votes:

            st.success(
                f"🤝 Majority prediction: "
                f"😏 Sarcastic "
                f"({sarcastic_votes}/{total_models} models)"
            )


        elif not_sarcastic_votes > sarcastic_votes:

            st.info(
                f"🤝 Majority prediction: "
                f"🙂 Not Sarcastic "
                f"({not_sarcastic_votes}/{total_models} models)"
            )


        else:

            st.warning(
                "⚖️ Models are evenly split: "
                f"{sarcastic_votes} Sarcastic vs "
                f"{not_sarcastic_votes} Not Sarcastic"
            )


# ============================================================
# 23. PERFORMANCE SECTION
# ============================================================

st.markdown("---")

chart_col, table_col = st.columns(
    [1.25, 1],
    gap="large"
)


# ============================================================
# 24. PERFORMANCE CHART
# ============================================================

with chart_col:

    st.markdown(
        "### 📊 Model Performance Comparison"
    )

    chart_data = (
        performance_data[
            [
                "Model",
                "Test Accuracy",
                "Precision",
                "Recall",
                "F1 Score"
            ]
        ]
        .set_index("Model")
        * 100
    )

    st.bar_chart(
        chart_data,
        use_container_width=True
    )

    st.success(
        "🏆 The pretrained Transformer achieved "
        "the strongest evaluation performance."
    )


# ============================================================
# 25. PERFORMANCE SUMMARY TABLE
# ============================================================

with table_col:

    st.markdown(
        "### 🏆 Performance Summary"
    )

    summary = performance_data[
        [
            "Model",
            "Test Accuracy",
            "Precision",
            "Recall",
            "F1 Score"
        ]
    ].copy()


    # ========================================================
    # CONVERT METRICS TO PERCENTAGES
    # ========================================================

    for column in [
        "Test Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]:

        summary[column] = summary[column].apply(
            lambda x:
            f"{x * 100:.2f}%"
            if pd.notna(x)
            else "N/A"
        )


    # ========================================================
    # DISPLAY TABLE
    # ========================================================

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # KEY INSIGHTS
    # ========================================================

    st.markdown(
        """
        #### 🔑 Key Insights

        - **Pretrained Transformer** achieved the highest
          test accuracy: **94.00%**.
        - It also achieved the highest available F1 score:
          **93.52%**.
        - **Tuned Bi-LSTM** achieved **86.18%** test accuracy.
        - **DistilBERT + LoRA** achieved **85.93%** accuracy
          while training only a small fraction of the model.
        - **TF-IDF + Linear SVM** provides a lightweight
          traditional ML baseline.
        """
    )


# ============================================================
# 26. DETAILED MODEL COMPARISON
# ============================================================

st.markdown("---")

st.markdown(
    "## 🔬 Detailed Model Comparison"
)

details = pd.DataFrame(
    {
        "Model": [
            "TF-IDF + Linear SVM",
            "Tuned Bi-LSTM",
            "DistilBERT + LoRA",
            "Pretrained Transformer"
        ],

        "Category": [
            "Machine Learning",
            "Deep Learning",
            "Transformer / PEFT",
            "Transformer"
        ],

        "Representation": [
            "TF-IDF",
            "Learned Embeddings",
            "DistilBERT",
            "Pretrained Transformer"
        ],

        "Training Strategy": [
            "Tuned Linear SVM",
            "Tuned Bi-LSTM",
            "LoRA Fine-Tuning",
            "Evaluation Only"
        ],

        "Test Accuracy": [
            "79.39%",
            "86.18%",
            "85.93%",
            "94.00%"
        ],

        "Recall": [
            "77.28%",
            "87.13%",
            "84.66%",
            "91.00%"
        ],

        "F1 Score": [
            "N/A",
            "N/A",
            "85.13%",
            "93.52%"
        ]
    }
)

st.dataframe(
    details,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 27. MODEL INFORMATION
# ============================================================

st.markdown("---")

st.markdown(
    "## 🧠 About the Models"
)

m1, m2 = st.columns(2)

with m1:

    with st.expander(
        "1️⃣ TF-IDF + Linear SVM"
    ):

        st.write(
            """
            **Type:** Traditional Machine Learning

            **Feature extraction:** TF-IDF

            **Classifier:** Tuned Linear SVM

            **Saved pipeline:** `tuned_tfidf_linear_svm.pkl`

            **Test Accuracy:** 79.39%

            **Recall:** 77.28%

            The saved sklearn Pipeline contains both the
            TF-IDF transformation and the tuned classifier.
            """
        )

    with st.expander(
        "2️⃣ Tuned Bi-LSTM"
    ):

        st.write(
            """
            **Type:** Deep Learning

            **Architecture:** Bidirectional LSTM

            **Model:** `tuned_bilstm.keras`

            **Tokenizer:** `tokenizer_dl.pkl`

            **Vocabulary:** 25,000

            **Sequence length:** 16

            **Padding:** post

            **Truncation:** post

            **Output:** sigmoid

            **Threshold:** 0.5

            **Test Accuracy:** 86.18%

            **Recall:** 87.13%
            """
        )


with m2:

    with st.expander(
        "3️⃣ DistilBERT + LoRA"
    ):

        st.write(
            """
            **Base model:** `distilbert-base-uncased`

            **Fine-tuning:** LoRA / PEFT

            **LoRA rank:** 8

            **LoRA alpha:** 16

            **LoRA dropout:** 0.1

            **Target modules:** `q_lin`, `v_lin`

            **Trainable parameters:** 739,586

            **Trainable percentage:** 1.09%

            **Test Accuracy:** 85.93%

            **Precision:** 85.61%

            **Recall:** 84.66%

            **F1 Score:** 85.13%
            """
        )

    with st.expander(
        "4️⃣ Pretrained Transformer"
    ):

        st.write(
            """
            **Model:** `helinivan/english-sarcasm-detector`

            **Source:** Hugging Face

            **Strategy:** Evaluation only

            **Additional training in this project:** None

            **Test Accuracy:** 94.00%

            **Precision:** 96.18%

            **Recall:** 91.00%

            **F1 Score:** 93.52%
            """
        )


# ============================================================
# 28. DATASET INFORMATION
# ============================================================

st.markdown("---")

st.markdown(
    "## 📚 Dataset Information"
)

d1, d2, d3 = st.columns(3)

with d1:
    st.metric(
        "Total Headlines",
        "25,713"
    )

with d2:
    st.metric(
        "Task",
        "Binary Classification"
    )

with d3:
    st.metric(
        "Classes",
        "2"
    )

st.write(
    """
    **Class 0:** Not Sarcastic

    **Class 1:** Sarcastic
    """
)


# ============================================================
# 29. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Predictions are model-generated and may not always "
    "correctly identify sarcasm, which can depend on context, "
    "tone and linguistic cues."
)
