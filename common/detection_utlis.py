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
def run_detection(file_input):
    """
    Runs intrusion detection on uploaded CSV file.
    file_input can be:
    - Streamlit UploadedFile
    - OR raw bytes
    Returns dataframe with predictions and confidence.
    """

    # ---------------------------------
    # Handle UploadedFile OR bytes
    # ---------------------------------
    if isinstance(file_input, bytes):
        file_bytes = file_input
    else:
        file_bytes = file_input.getvalue()

    # -------- Save file temporarily --------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    # -------- Read raw data --------
    df = pd.read_csv(temp_path)

    # -------- Preprocess --------
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
    preds = model.predict(X, verbose=0)
    pred_classes = np.argmax(preds, axis=1)
    pred_labels = label_encoder.inverse_transform(pred_classes)

    # -------- Confidence --------
    confidence = np.max(preds, axis=1)

    # -------- Add results --------
    df["Predicted_Attack"] = pred_labels
    df["Confidence"] = confidence.round(3)
    df["Severity"] = df["Predicted_Attack"].apply(detect_severity)

    return df

import time

def stream_detection(file_input, delay=1, start_index=0):
    """
    file_input can be:
    - Streamlit UploadedFile
    - OR raw bytes (from session_state)
    """

    import tempfile
    import pandas as pd
    import numpy as np
    import time
    import os

    # ---------------------------------
    # Handle UploadedFile OR bytes
    # ---------------------------------
    if isinstance(file_input, bytes):
        file_bytes = file_input
    else:
        file_bytes = file_input.getvalue()

    # Save once to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    # Read CSV
    df = pd.read_csv(temp_path)

    # Preprocess once
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

    # ---------------------------------
    # Stream row by row
    # ---------------------------------
    for i in range(start_index, len(X)):
        row_X = X[i:i+1]

        preds = model.predict(row_X, verbose=0)
        pred_class = np.argmax(preds, axis=1)
        pred_label = label_encoder.inverse_transform(pred_class)[0]
        confidence = float(np.max(preds))

        yield {
            "row": i + 1,
            "prediction": pred_label,
            "confidence": round(confidence, 3),
            "severity": detect_severity(pred_label)
        }

        time.sleep(delay)
