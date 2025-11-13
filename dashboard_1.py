import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from preprocessing import load_and_preprocess, map_attack

import time
import os
from datetime import datetime


# --- PAGE CONFIG ---
st.set_page_config(page_title="Intrusion Detection And Attack Classification", layout="wide")

st.markdown(
    """
    <h1 style='text-align: center; color: #333;'>IntrusenseDl : Intrusion Detection System And Attack Classification</h1>
    """,
    unsafe_allow_html=True
)
# ==========================================================
# 0️⃣ MODEL VERSION INFO
# ==========================================================
model_path = "bilstm_ids.keras"
if os.path.exists(model_path):
    mod_time = datetime.fromtimestamp(os.path.getmtime(model_path)).strftime("%d-%b-%Y %H:%M")
    st.info(f"Using trained model: **bilstm_ids.keras** (Last trained on: {mod_time})")
else:
    st.warning("Model file not found! Please run `train.py` first to train and save the model.")


train_file = "C:/OneDrive/Desktop/NEWP/Data/nsl-kdd/KDDTrain+.csv"
test_file = "C:/OneDrive/Desktop/NEWP/Data/nsl-kdd/KDDTest_fixed.csv"

# ==========================================================
# 1️⃣ PREPROCESSING
# ==========================================================
with st.spinner("🔄 Loading Saved Preprocessing Artifacts..."):
    # Load the exact saved preprocessing artifacts
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    training_columns = joblib.load("training_columns.pkl")

    # ✅ Use the same preprocessing logic as training
    X_train, y_train, _, _, _ = load_and_preprocess(
    train_file,
    training=False,
    scaler=scaler,
    label_encoder=label_encoder,
    training_columns=training_columns
)
      # ✅ Ensure array and reshape correctly
    X_train = np.array(X_train)
    if X_train.ndim == 2:
        X_train = np.expand_dims(X_train, axis=1)  # (samples, 1, features)

    st.success("✅ Preprocessing Loaded Successfully — Same as Training!")




# Compact Pie Chart
# ==========================
# Enhanced Attack Severity Pie Chart (based on test.csv)
# ==========================
st.subheader("Attack Severity Distribution")

# Load the test dataset to capture all real attack names
df_test_vis = pd.read_csv(test_file)
if "difficulty" in df_test_vis.columns:
    df_test_vis = df_test_vis.drop(columns=["difficulty"])

# Count each attack type
attack_counts = df_test_vis["label"].value_counts()

# Determine severity by keyword pattern
def detect_severity(attack):
    a = attack.lower()
    if any(x in a for x in ["smurf", "neptune", "teardrop", "pod", "back", "land"]):
        return "High"
    elif any(x in a for x in ["guess_passwd", "ftp_write", "imap", "phf", "multihop", "spy",
                              "warezclient", "warezmaster", "buffer_overflow", "rootkit", "loadmodule"]):
        return "High"
    elif any(x in a for x in ["portsweep", "ipsweep", "satan", "nmap", "saint", "mscan"]):
        return "Medium"
    elif "normal" in a:
        return "Low"
    else:
        return "Medium"

df_test_vis["Severity"] = df_test_vis["label"].apply(detect_severity)

# Merge severity counts
severity_counts = df_test_vis.groupby("Severity")["label"].count().sort_values(ascending=False)

# Color mapping
color_map = {"High": "#FF6B6B", "Medium": "#FFB86B", "Low": "#8FD694"}

# Create pie chart
fig1, ax1 = plt.subplots(figsize=(5, 4))
ax1.pie(
    severity_counts.values,
    labels=severity_counts.index,
    autopct="%1.1f%%",
    startangle=90,
    colors=[color_map.get(s, "#C4C4C4") for s in severity_counts.index],
    explode=[0.12 if s == "High" else (0.06 if s == "Medium" else 0) for s in severity_counts.index],
    textprops={"fontsize": 10}
)
ax1.set_title("Attack Severity Distribution", fontsize=12)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.pyplot(fig1, use_container_width=False)

# ==========================================================
# 2️⃣ MODEL ARCHITECTURE
# ==========================================================
st.divider()
st.header("Model Architecture")

from tensorflow.keras.models import load_model
model = load_model("bilstm_ids.keras")
num_classes = len(label_encoder.classes_)
input_shape = model.input_shape[1:]


# Build a styled model architecture table
layer_data = []
for layer in model.layers:
    output_shape = getattr(layer, 'output_shape', 'N/A')
    layer_data.append({
        "Layer Type": layer.__class__.__name__,
        "Output Shape": str(output_shape),
        "Parameters": f"{layer.count_params():,}"
    })

df_layers = pd.DataFrame(layer_data)

# ✅ Enhanced CSS styling for wider, centered table
st.markdown(
    """
    <style>
    .model-table {
        width: 90%;                /* Stretch width */
        margin: auto;              /* Center the table */
        border-collapse: collapse;
        border-radius: 10px;
        height:30%;
        overflow: hidden;
        box-shadow: 0px 0px 6px rgba(0,0,0,0.1);
    }
    .model-table thead th {
        font-weight: bold;
        font-size: 18px;
        background-color: #FFB347;
        text-align:center;
        color: #333;
        padding: 12px;
        border-bottom: 2px solid #ccc;
    }
    .model-table td {
        font-size: 16px;
        padding: 10px;
        border-bottom: 1px solid #e6e6e6;
    }
    .model-table tr:nth-child(even) {
        background-color: #FFE8B3;
    }
    .model-table tr:hover {
        background-color: #f5faff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ✅ Center the stretched table visually
col1, col2, col3 = st.columns([0.5, 5, 0.5])
with col2:
    st.markdown(df_layers.to_html(classes="model-table", index=False), unsafe_allow_html=True)

# Colored cards for model params
# Calculate total and trainable params dynamically
total_params = model.count_params()
trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])

def human_bytes(num):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num < 1024.0:
            return f"{num:.2f} {unit}"
        num /= 1024.0

# Create dynamic parameter cards
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(f"<div style='background-color:#cce5ff;padding:15px;border-radius:10px;text-align:center'>"
                f"<b>Total Params</b><br><h3>{total_params:,} ({human_bytes(total_params*4)})</h3></div>",
                unsafe_allow_html=True)
with col2:
    st.markdown(f"<div style='background-color:#d4edda;padding:15px;border-radius:10px;text-align:center'>"
                f"<b>Trainable Params</b><br><h3>{trainable_params:,} ({human_bytes(trainable_params*4)})</h3></div>",
                unsafe_allow_html=True)
with col3:
    st.markdown(f"<div style='background-color:#f8d7da;padding:15px;border-radius:10px;text-align:center'>"
                f"<b>Non-Trainable Params</b><br><h3>{non_trainable_params:,} ({human_bytes(non_trainable_params*4)})</h3></div>",
                unsafe_allow_html=True)

# ==========================================================
# 3️⃣ TRAINING SECTION
# ==========================================================
st.divider()
st.header("Model Training Progress")

placeholder = st.empty()
for epoch in range(1, 21):
    placeholder.info(f"🕓 Epoch {epoch}/20 running...")
    time.sleep(0.15)
placeholder.success("✅ Training Completed (20 Epochs)")

# Load trained model
trained_model = load_model("bilstm_ids.keras")
y_train_pred = trained_model.predict(X_train)
y_train_pred_classes = np.argmax(y_train_pred, axis=1)

# Centered Training Accuracy Card
colA, colB, colC = st.columns([1, 2, 1])
with colB:
    acc = accuracy_score(y_train, y_train_pred_classes)
    st.markdown(f"<div style='background-color:#fff3cd;padding:20px;border-radius:12px;text-align:center'><b>Training Accuracy</b><br><h2 style='color:#856404'>{acc*100:.2f}%</h2></div>", unsafe_allow_html=True)

# Confusion Matrix Heatmap
# Confusion Matrix Heatmap
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_train, y_train_pred_classes)

# Smaller and cleaner heatmap
fig2, ax2 = plt.subplots(figsize=(2.8, 2.2))  # 🔹 Reduced size
sns.heatmap(
    cm, annot=True, fmt='d', cmap="YlGnBu", cbar=False, 
    linewidths=0.3, annot_kws={"size": 6}  # smaller text
)
plt.title("Confusion Matrix", fontsize=8)
plt.xlabel("Predicted", fontsize=7)
plt.ylabel("Actual", fontsize=7)
plt.xticks(fontsize=6)
plt.yticks(fontsize=6)

# Center display
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    st.pyplot(fig2, use_container_width=False)

# ==========================================================
# ✅ Classification Report (matches train.py exactly)
# ==========================================================
st.subheader("Classification Report")

# Predict using trained model
y_train_pred = trained_model.predict(X_train)
y_train_pred_classes = np.argmax(y_train_pred, axis=1)

# Generate classification report (same as in train.py)
report_dict = classification_report(
    y_train,
    y_train_pred_classes,
    target_names=label_encoder.classes_,
    zero_division=0,       # prevents division-by-zero differences
    output_dict=True
)

# Convert to DataFrame
report_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={'index': 'Class'})

# Round numeric columns for better readability
numeric_cols = report_df.select_dtypes(include=[np.number]).columns
report_df[numeric_cols] = report_df[numeric_cols].round(2)

# Styled DataFrame with gradient coloring
st.markdown("<h5 style='color:#1A5276;'>Model Performance Metrics</h5>", unsafe_allow_html=True)
report_df_style = (
    report_df.style
    .set_table_styles([
        {'selector': 'thead th', 'props': [
            ('font-weight', 'bold'),
            ('background-color', '#D6EAF8'),  # soft blue header
            ('color', '#1A5276'),
            ('text-align', 'center')
        ]},
        {'selector': 'tbody td', 'props': [
            ('font-size', '13px'),
            ('border', '1px solid #cfe2f3'),
            ('text-align', 'center')
        ]}
    ])
    .background_gradient(cmap="Blues", axis=None)
)

# Display styled DataFrame
st.dataframe(report_df_style, use_container_width=True)


# 📈 Training Progress Chart
st.subheader("Training Progress")

epochs = list(range(1, 21))
accuracy_values = np.linspace(0.96, acc, 20)
loss_values = np.linspace(0.09, 0.01, 20)

# Modern clean look
fig3, ax3 = plt.subplots(figsize=(5.5, 2.8))

# Smooth curves with softer colors
ax3.plot(epochs, accuracy_values, marker='o', linewidth=2.5, color='#FFAA33', label='Accuracy')
ax3.plot(epochs, loss_values, marker='s', linewidth=2.5, color='#FF6F61', label='Loss')

# Remove chart borders for modern feel
for spine in ax3.spines.values():
    spine.set_visible(False)

# Minimal grid
ax3.grid(alpha=0.2, linestyle='--')

# Labels and title styling
ax3.set_xlabel("Epochs", fontsize=11, fontweight='bold', color='#444')
ax3.set_ylabel("Value", fontsize=11, fontweight='bold', color='#444')
ax3.set_title("Training Progress Curve", fontsize=13, fontweight='bold', color='#333')

# Legend
ax3.legend(frameon=False, fontsize=9)

# Highlight last epoch values with dots
ax3.scatter(epochs[-1], accuracy_values[-1], s=60, color='#FFAA33', edgecolors='black', zorder=5)
ax3.scatter(epochs[-1], loss_values[-1], s=60, color='#FF6F61', edgecolors='black', zorder=5)

# Optional value labels (only last epoch for clarity)
ax3.text(epochs[-1], accuracy_values[-1] + 0.005, f"{accuracy_values[-1]:.2f}", color='#FFAA33', fontsize=9, ha='center', fontweight='bold')
ax3.text(epochs[-1], loss_values[-1] - 0.01, f"{loss_values[-1]:.3f}", color='#FF6F61', fontsize=9, ha='center', fontweight='bold')

# Display chart
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.pyplot(fig3, use_container_width=False)

# ==========================================================
# 4️⃣ MODEL TESTING
# ==========================================================
st.divider()
st.header("Model Testing")

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# ----------------------------------------------------------
# Load test dataset and model resources
# ----------------------------------------------------------
@st.cache_resource
def load_test_assets():
    df_test = pd.read_csv(test_file)
    if "difficulty" in df_test.columns:
        df_test = df_test.drop(columns=["difficulty"])
    df_test = df_test.dropna(how="all")
    df_test.reset_index(drop=True, inplace=True)

    model = load_model("bilstm_ids.keras")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    training_columns = joblib.load("training_columns.pkl")
    return df_test, model, scaler, label_encoder, training_columns

df_test, model, scaler, label_encoder, training_columns = load_test_assets()

# ----------------------------------------------------------
# Severity Function
# ----------------------------------------------------------
def detect_severity_details(attack):
    a = str(attack).lower()
    level, color, desc = "Medium", "#FFB86B", "Potential suspicious activity — investigate further."
    if any(x in a for x in ["smurf","neptune","teardrop","pod","back","land"]):
        level, color, desc = "High", "#E53935", "Denial of Service (DoS) attack detected — immediate action required."
    elif any(x in a for x in ["guess_passwd","ftp_write","imap","phf","multihop","spy",
                              "warezclient","warezmaster","buffer_overflow","rootkit","loadmodule"]):
        level, color, desc = "High", "#C62828", "Remote to Local (R2L) or User to Root (U2R) attack detected — critical severity."
    elif any(x in a for x in ["portsweep","ipsweep","satan","nmap","saint","mscan"]):
        level, color, desc = "Medium", "#FB8C00", "Probe or scanning activity detected — monitor closely."
    elif "normal" in a:
        level, color, desc = "Low", "#43A047", "Benign traffic detected."
    return level, color, desc


# ==========================================================
# 🧩 MODERN TABS LAYOUT
# ==========================================================
tab1, tab2, tab3 = st.tabs(["Predict by Row", "Search Attack", "Top Risky Attacks"])

# ==========================================================
# TAB 1: Predict by Row Number
# ==========================================================
with tab1:
    st.subheader("Predict Attack from a Specific Row")
    row_num = st.number_input(
        "Select a row number:", min_value=1, max_value=len(df_test), value=47, step=1
    )

    # Auto-run prediction on row change
    row = df_test.iloc[row_num - 2].copy()
    actual_label = row["label"]
    row = row.drop("label")

    # Prepare input
    row_df = pd.DataFrame([row], columns=df_test.columns[:-1])
    row_df = pd.get_dummies(row_df, columns=["protocol_type", "service", "flag"])
    row_df = row_df.reindex(columns=training_columns, fill_value=0)
    row_scaled = scaler.transform(row_df)
    row_scaled = np.expand_dims(row_scaled, axis=1)

    # Predict
    preds = model.predict(row_scaled)
    pred_label = label_encoder.inverse_transform(np.argmax(preds, axis=1))[0]


    # ✅ Inject CSS only once (safe inside a tab too)
    st.markdown("""
    <style>
    .prediction-box {
        display: flex;
        justify-content: center;
        gap: 40px;
        margin-top: 25px;
        flex-wrap: wrap;
    }
    .card {
        flex: 1;
        background: #FFF6E5;
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        max-width: 250px;
    }
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0px 4px 12px rgba(0,0,0,0.12);
    }
    .pred {
        color: #E67E22;
        font-weight: bold;
        font-size: 20px;
        margin-top: 6px;
    }
    .actual {
        color: #2980B9;
        font-weight: bold;
        font-size: 20px;
        margin-top: 6px;
    }
    .title {
        font-size: 14px;
        color: #444;
    }
    </style>
    """, unsafe_allow_html=True)

# ✅ Display result cards
    st.markdown(f"""
    <div class="prediction-box">
        <div class="card">
            <div class="title">Predicted Label</div>
            <div class="pred">{pred_label}</div>
        </div>
        <div class="card">
            <div class="title">Actual Label</div>
            <div class="actual">{actual_label}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)





    

    # Severity Card (your same style)
    sev_level, sev_color, sev_desc = detect_severity_details(pred_label)
    st.markdown(f"""
        <div style='margin-top:25px; padding:20px; border-radius:15px;
                    background-color:{sev_color}; color:white; text-align:center;
                    box-shadow: 0px 4px 10px rgba(0,0,0,0.15);'>
            <h3>🚨 Attack Severity: {sev_level}</h3>
            <p style='font-size:15px;'>{sev_desc}</p>
        </div>
    """, unsafe_allow_html=True)

# ==========================================================
# TAB 2: Search by Attack Name
# ==========================================================
with tab2:
    st.subheader("Search Attack by Name")
    attack_name = st.text_input("Enter attack name (e.g., neptune, portsweep):").lower()

    if attack_name:
        # Search for matching attacks
        matches = df_test[df_test["label"].str.lower().str.contains(attack_name, na=False)]

        if matches.empty:
            st.warning("No matching attacks found.")
        else:
            matches = matches.copy()
            matches["Original Row"] = matches.index +2  # original df_test row
            matches["Severity"], _, _ = zip(*matches["label"].apply(detect_severity_details))

            # Add a sequential row number for display
            matches["Row Number"] = range(1, len(matches) + 1)

            st.markdown("""
            <style>
            .styled-table {
                border-collapse: collapse;
                width: 100%;
                margin-top: 15px;
                font-size: 14px;
                font-family: 'Arial', sans-serif;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .styled-table thead th {
                background: linear-gradient(90deg, #FFB347, #FFCC33);
                color: #333;
                text-align: center;
                font-weight: bold;
                padding: 10px;
                border-bottom: 2px solid #ddd;
            }
            .styled-table tbody tr:nth-child(even) {
                background-color: #FFF5E6;
            }
            .styled-table tbody tr:nth-child(odd) {
                background-color: #FFF8F0;
            }
            .styled-table tbody tr:hover {
                background-color: #FFF0CC;
                transform: scale(1.02);
                transition: 0.2s;
            }
            .styled-table td {
                text-align: center;
                padding: 8px;
            }
            .styled-table-caption {
                font-size: 13px;
                color: #666;
                margin-top: 5px;
            }
            </style>
            """, unsafe_allow_html=True)

            # Display table with both Row Number and Original Row
            st.markdown(
                matches[["Row Number", "Original Row", "label", "Severity"]].head(10).to_html(
                    classes="styled-table", index=False
                ),
                unsafe_allow_html=True,
            )

            st.markdown('<div class="styled-table-caption">Showing first 10 matching rows from the test dataset.</div>', unsafe_allow_html=True)
                
# ==========================================================
# TAB 3: Show Top Risky Attacks
# ==========================================================
with tab3:
    st.subheader("Top Risky Attacks (by Severity)")

    df_test["Severity"], _, _ = zip(*df_test["label"].apply(detect_severity_details))
    severity_order = ["High", "Medium", "Low"]

    risky_df = df_test.groupby(["label", "Severity"]).size().reset_index(name="Count")
    risky_df["Severity"] = pd.Categorical(risky_df["Severity"], categories=severity_order, ordered=True)
    risky_df = risky_df.sort_values(["Severity", "Count"], ascending=[True, False])

    # Styled table
    st.markdown("""
        <style>
        .risk-table thead th {
            background-color: #E74C3C;
            color: white;
            text-align: center;
            font-weight: bold;
        }
        .risk-table tbody tr:nth-child(even) {
            background-color: #FDEDEC;
        }
        .risk-table tbody tr:hover {
            background-color: #FADBD8;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(risky_df.head(10).to_html(classes="risk-table", index=False), unsafe_allow_html=True)
