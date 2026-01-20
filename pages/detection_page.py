import streamlit as st
import pandas as pd
import time

from common.session import require_login
from common.detection_utils import run_detection, detect_severity, stream_detection

# ---------------- AUTH ----------------
require_login()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Intrusion Detection", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
.card {
    background: linear-gradient(135deg, #1f2937, #111827);
    padding: 18px;
    border-radius: 16px;
    color: white;
    text-align: center;
    box-shadow: 0 0 15px rgba(0,255,255,0.12);
}
.sub { color: #9ca3af; }
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("Intrusion Detection System")
st.caption("Live and offline intrusion detection dashboard")

# ---------------- SESSION STATE INIT ----------------
if "live_running" not in st.session_state:
    st.session_state["live_running"] = False

if "live_log" not in st.session_state:
    st.session_state["live_log"] = []

# =====================================================
# UPLOAD DATASET
# =====================================================
st.subheader("Upload Network Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV file (network traffic)",
    type=["csv"]
)

if uploaded_file:
    st.success("Dataset uploaded successfully")

    if st.button("Run Detection"):
        with st.spinner("Running intrusion detection..."):
            result_df = run_detection(uploaded_file)

        st.session_state["last_detection"] = result_df
        st.success("Detection completed successfully")

# =====================================================
# RESULTS SECTION
# =====================================================
if "last_detection" in st.session_state:

    df = st.session_state["last_detection"]

    # ---------------- KPI CARDS ----------------
    total_records = len(df)
    total_attacks = len(df[df["Predicted_Attack"] != "Normal"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='card'><h3>{total_records}</h3><div class='sub'>Total Records</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='card'><h3>{total_attacks}</h3><div class='sub'>Detected Attacks</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='card'><h3>{round((total_attacks/total_records)*100,2)}%</h3><div class='sub'>Attack Ratio</div></div>", unsafe_allow_html=True)

    st.divider()

    # =====================================================
    # TABS
    # =====================================================
    tab_live, tab_row, tab_search, tab_severity, tab_results = st.tabs([
        "Live Stream",
        "Predict by Row",
        "Search Attack",
        "Severity View",
        "Full Results"
    ])

    # =====================================================
    # LIVE STREAM
    # =====================================================
    with tab_live:
        st.subheader("Live Intrusion Detection")

        colA, colB = st.columns(2)

        with colA:
            if st.button("Start"):
                st.session_state["live_running"] = True

        with colB:
            if st.button("Stop"):
                st.session_state["live_running"] = False

        placeholder = st.empty()

        def color_row(row):
            if row["Severity"] == "Low":
                return ["background-color: #558c1b; color: white"] * len(row)
            elif row["Severity"] == "Medium":
                return ["background-color: #f59764; color: black"] * len(row)
            else:
                return ["background-color: #d63c3c; color: white"] * len(row)

        if uploaded_file and st.session_state["live_running"]:
            for event in stream_detection(uploaded_file, delay=2):

                if not st.session_state["live_running"]:
                    st.warning("Live stream stopped")
                    break

                st.session_state["live_log"].append({
                    "Row": event["row"],
                    "Predicted_Attack": event["prediction"],
                    "Confidence": event["confidence"],
                    "Severity": event["severity"]
                })

                live_df = pd.DataFrame(st.session_state["live_log"])
                placeholder.dataframe(live_df.style.apply(color_row, axis=1), use_container_width=True)

        elif st.session_state["live_log"]:
            live_df = pd.DataFrame(st.session_state["live_log"])
            placeholder.dataframe(live_df.style.apply(color_row, axis=1), use_container_width=True)

    # =====================================================
    # PREDICT BY ROW
    # =====================================================
    with tab_row:
        st.subheader("Predict by Row Number")

        row_num = st.number_input(
            "Enter row number",
            min_value=1,
            max_value=len(df),
            value=1,
            step=1
        )

        row = df.iloc[row_num]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='card'><h4>Predicted Attack</h4><h2>{row['Predicted_Attack']}</h2></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='card'><h4>Severity</h4><h2>{detect_severity(row['Predicted_Attack'])}</h2></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='card'><h4>Confidence</h4><h2>{round(row['Confidence']*100,2)}%</h2></div>", unsafe_allow_html=True)

    # =====================================================
    # SEARCH ATTACK
    # =====================================================
    with tab_search:
        st.subheader("Search by Attack Name")

        query = st.text_input("Enter attack name").lower()

        if query:
            filtered = df[df["Predicted_Attack"].str.lower().str.contains(query)]

            if filtered.empty:
                st.warning("No matching attacks found")
            else:
                filtered = filtered.copy()
                filtered["Row"] = filtered.index + 1
                st.dataframe(filtered[["Row", "Predicted_Attack", "Confidence"]], use_container_width=True)

    # =====================================================
    # SEVERITY VIEW
    # =====================================================
    with tab_severity:
        st.subheader("Attack Severity Overview")

        df["Severity"] = df["Predicted_Attack"].apply(detect_severity)
        st.bar_chart(df["Severity"].value_counts())
        st.dataframe(df[["Predicted_Attack", "Severity", "Confidence"]], use_container_width=True)

    # =====================================================
    # FULL RESULTS
    # =====================================================
    with tab_results:
        st.subheader("Detection Results (Offline)")
        st.dataframe(df, use_container_width=True)

else:
    st.info("Upload a dataset and run detection to see results.")
