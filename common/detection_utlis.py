import pandas as pd
import numpy as np
import joblib
import tempfile
import os

from ml.preprocessing import load_and_preprocess
from ml.model import build_bilstm

# =================================================
# LOAD PREPROCESSING OBJECTS
# =================================================
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")
training_columns = joblib.load("training_columns.pkl")

num_classes = len(label_encoder.classes_)

# =================================================
# BUILD & LOAD MODEL
# =================================================
input_shape = (1, len(training_columns))
model = build_bilstm(input_shape, num_classes)
model.load_weights("bilstm_ids.weights.h5")

# =================================================
# SEVERITY LOGIC
# =================================================
def detect_severity(attack):
    attack = str(attack).lower()

    if attack == "normal":
        return "Low"

    elif attack in ["probe"]:
        return "Medium"

    elif attack in ["dos", "r2l", "u2r"]:
        return "High"

    else:
        return "Medium"


# =================================================
# MAIN DETECTION FUNCTION
# =================================================
def run_detection(uploaded_file):
    """
    Runs intrusion detection on uploaded CSV file.
    Returns dataframe with predictions and confidence.
    """

    # -------- Save uploaded file temporarily --------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name

    # -------- Read raw data --------
    df = pd.read_csv(temp_path)

    # -------- Preprocess (same as training) --------
    X, _, _, _, _ = load_and_preprocess(
        filepath=temp_path,
        training=False,
        has_header=True,
        scaler=scaler,
        label_encoder=label_encoder,
        training_columns=training_columns
    )

    os.remove(temp_path)

    # -------- Reshape for BiLSTM --------
    if X.ndim == 2:
        X = np.expand_dims(X, axis=1)

    # -------- Prediction --------
    preds = model.predict(X)
    pred_classes = np.argmax(preds, axis=1)
    pred_labels = label_encoder.inverse_transform(pred_classes)

    # -------- Confidence (max probability) --------
    confidence = np.max(preds, axis=1)

    # -------- Add results --------
    df["Predicted_Attack"] = pred_labels
    df["Confidence"] = confidence.round(3)
    df["Severity"] = df["Predicted_Attack"].apply(detect_severity)

    return df

import time

def stream_detection(uploaded_file, delay=1):
    """
    Simulates real-time intrusion detection
    Yields one prediction at a time
    """

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.getvalue())
        temp_path = tmp.name

    df = pd.read_csv(temp_path)

    # Preprocess full dataset once
    X, _, _, _, _ = load_and_preprocess(
        filepath=temp_path,
        training=False,
        has_header=True,
        scaler=scaler,
        label_encoder=label_encoder,
        training_columns=training_columns
    )

    os.remove(temp_path)

    if X.ndim == 2:
        X = np.expand_dims(X, axis=1)

    # Predict row-by-row
    for i in range(len(X)):
        row_X = X[i:i+1]
        raw_row = df.iloc[i].copy()

        preds = model.predict(row_X, verbose=0)
        pred_class = np.argmax(preds, axis=1)
        pred_label = label_encoder.inverse_transform(pred_class)[0]
        confidence = float(np.max(preds))

        yield {
            "row": i + 1,
            "prediction": pred_label,
            "confidence": round(confidence, 3),
            "severity": detect_severity(pred_label),
            "data": raw_row
        }

        time.sleep(delay)
