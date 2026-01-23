import streamlit as st
import pandas as pd
import time
import altair as alt
from streamlit_autorefresh import st_autorefresh

from common.session import require_login
from common.detection_utils import run_detection, detect_severity, stream_detection

# ---------------- AUTH ----------------
require_login()

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Intrusion Detection", layout="wide")

# ---------------- GLOBAL CSS ----------------
st.markdown("""
<style>

/* ---------- HEADER ---------- */
.header {
    position: fixed;
    top: 3.5rem;
    left: 0;
    right: 0;
    height: 80px;
    background: linear-gradient(90deg, #020617, #0f172a);
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;   /* 👈 CENTER CONTENT */
    text-align: center;        /* 👈 CENTER TEXT */
    z-index: 1000;
    box-shadow: 0 4px 18px rgba(0,0,0,0.3);
}


.header-title {
    font-size: 20px;
    font-weight: 700;
}

.header-subtitle {
    font-size: 13px;
    color: #cbd5f5;
    margin-top: 2px;
}

/* ---------- PAGE SPACING ---------- */
.block-container {
    padding-top: 150px;
    padding-bottom: 90px;
}

/* ---------- FOOTER ---------- */
.footer {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    height: 60px;
    background: #020617;
    color: #94a3b8;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 13px;
    z-index: 1000;
}
</style>
""", unsafe_allow_html=True)

# ---------------- CSS ----------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{
    background:#ffffff;
}
.card{
    background:#ffffff;
    padding:18px;
    border-radius:16px;
    text-align:center;
    border:1px solid #d1fae5;
    box-shadow:0 8px 20px rgba(0,0,0,0.06);
}
</style>
""", unsafe_allow_html=True)


# ---------------- SESSION STATE INIT ----------------
if "live_running" not in st.session_state:
    st.session_state.live_running = False
if "live_index" not in st.session_state:
    st.session_state.live_index = 0
if "live_generator" not in st.session_state:
    st.session_state.live_generator = None
if "chart_data" not in st.session_state:
    st.session_state.chart_data = pd.DataFrame(
        columns=["packet", "severity", "attack", "confidence"]
    )
if st.session_state.get("_last_page") != "detection":
    st.session_state.live_running = False
    st.session_state.live_generator = None

if "stream_busy" not in st.session_state:
    st.session_state.stream_busy = False

st.session_state["_last_page"] = "detection"

# ================= HEADER =================
st.markdown("""
<div class="header">
    <div>
        <div class="header-title">Intrusion Detection System</div>
        <div class="header-subtitle">Live and Offline Network Threat Monitoring</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ================= UPLOAD =================
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file and "uploaded_bytes" not in st.session_state:
    st.session_state.uploaded_bytes = uploaded_file.getvalue()

if uploaded_file and st.button("Start Detection"):
    st.session_state.last_detection = run_detection(
        st.session_state.uploaded_bytes
    )
    st.session_state.chart_data = st.session_state.chart_data.iloc[0:0]
    st.session_state.live_index = 0
    st.session_state.live_generator = None
    st.session_state.live_running = False
    st.success(" ")

# =====================================================
# MAIN CONTENT
# =====================================================
if "last_detection" in st.session_state:
    df = st.session_state.last_detection

    st.divider()

        # =====================================================
    # LIVE STREAM
    # =====================================================
    st.subheader("Live Intrusion Detection")
    chart_placeholder = st.container()

    colA, colB = st.columns(2)

    with colA:
        if st.button("Start Live Stream"):
            st.session_state.live_running = True
            st.session_state.live_generator = stream_detection(
                st.session_state.uploaded_bytes,
                start_index=st.session_state.live_index,
                delay=0
            )

    with colB:
        if st.button("Stop Live Stream"):
            st.session_state.live_running = False
            st.session_state.live_generator = None

    # Auto refresh every 1 second while live is running
    if st.session_state.live_running:
        st_autorefresh(interval=2000, key="live_refresh")


    def render_chart():
        if st.session_state.chart_data.empty:
            return

        base = alt.Chart(st.session_state.chart_data)
        line = base.mark_line(strokeWidth=3, color="#39ff14").encode(
            x=alt.X("packet:Q",  title="Packet Number", scale=alt.Scale(nice=False)),
            y=alt.Y(
                "severity:Q",
                scale=alt.Scale(domain=[0.5, 3.5]),
                axis=alt.Axis(
                    values=[1, 2, 3],
                    labelExpr="datum.value == 1 ? 'Low' : datum.value == 2 ? 'Medium' : 'High'"
                )
            )
        )

        points = base.mark_circle(size=90).encode(
            x="packet:Q",
            y="severity:Q",
            color="attack:N",
            tooltip=["packet", "attack", "confidence"]
        )

        chart_placeholder.altair_chart(
            (line + points).properties(height=380),
            use_container_width=True
        )


    # Stream one event per refresh
    if (
        st.session_state.live_running
        and st.session_state.live_generator
        and not st.session_state.stream_busy
    ):
        try:
            st.session_state.stream_busy = True

            event = next(st.session_state.live_generator)

            sev = 1 if event["severity"] == "Low" else 2 if event["severity"] == "Medium" else 3

            st.session_state.chart_data.loc[len(st.session_state.chart_data)] = {
                "packet": event["row"],
                "severity": sev,
                "attack": event["prediction"],
                "confidence": event["confidence"]
            }

            st.session_state.live_index = event["row"]

        except StopIteration:
            st.session_state.live_running = False
            st.session_state.live_generator = None

        finally:
            st.session_state.stream_busy = False

    # Always render chart
    render_chart()

    # =====================================================
    # LIVE DATA LIMIT
    # =====================================================
    if st.session_state.live_index > 0:
        live_df = df.iloc[:st.session_state.live_index]
    else:
        live_df = pd.DataFrame()


    # =====================================================
    # TABS (BELOW LIVE STREAM)
    # =====================================================

    # =====================================================
# SHOW OPTIONS ONLY WHEN LIVE STREAM IS STOPPED
# =====================================================
    if not st.session_state.live_running and not live_df.empty:

        st.divider()
        tab_row, tab_search, tab_severity, tab_results = st.tabs([
            "Predict by Packet Number",
            "Search Attack",
            "Severity View",
            "Full Results"
        ])

        # ---------------- PREDICT BY ROW ----------------
        with tab_row:
            st.subheader("Predict by Packet Number")

            if live_df.empty:
                st.info("Start live stream to see predictions")
            else:
                row_no = st.number_input(
                    "Select Packet Number",
                    min_value=1,
                    max_value=len(live_df),
                    value=1,
                    step=1
                )

                # Convert user-friendly (1-based) to pandas (0-based)
                row = live_df.iloc[row_no - 1]

                sev = detect_severity(row["Predicted_Attack"])

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown(f"""
                    <div class="card">
                        <h4>Predicted Attack</h4>
                        <h2>{row['Predicted_Attack']}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    st.markdown(f"""
                    <div class="card">
                        <h4>Severity</h4>
                        <h2>{sev}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div class="card">
                        <h4>Confidence</h4>
                        <h2>{round(row['Confidence'] * 100, 2)}%</h2>
                    </div>
                    """, unsafe_allow_html=True)



        # ---------------- SEARCH ATTACK ----------------
        with tab_search:
            st.subheader("Search Attack (Live Data Only)")

            if live_df.empty:
                st.info("No live data yet")
            else:
                query = st.text_input("Enter attack name").lower()

                if query:
                    result = live_df[
                        live_df["Predicted_Attack"].str.lower().str.contains(query)
                    ].copy()

                    if result.empty:
                        st.warning("No matching attacks found")
                    else:
                        result["Row"] = result.index + 1
                        result["Severity"] = result["Predicted_Attack"].apply(detect_severity)
                        result["Confidence (%)"] = (result["Confidence"] * 100).round(2)

                        st.dataframe(
                            result[["Row", "Predicted_Attack", "Severity", "Confidence (%)"]],
                            use_container_width=True
                        )

        # ---------------- SEVERITY VIEW ----------------
        with tab_severity:
            st.subheader("Severity Distribution (Live Data)")

            if live_df.empty:
                st.info("No live data yet")
            else:
                temp_df = live_df.copy()
                temp_df["Severity"] = temp_df["Predicted_Attack"].apply(detect_severity)

                severity_counts = temp_df["Severity"].value_counts().reset_index()
                severity_counts.columns = ["Severity", "Count"]

                pie = alt.Chart(severity_counts).mark_arc(innerRadius=60).encode(
                    theta="Count:Q",
                    color="Severity:N",
                    tooltip=["Severity", "Count"]
                ).properties(height=350)

                st.altair_chart(pie, use_container_width=True)


        # ---------------- FULL RESULTS ----------------
        with tab_results:
            st.subheader("Full Results (Live Data Only)")

            if live_df.empty:
                st.info("No live data yet")
            else:
                df_show = live_df.reset_index(drop=True)
                df_show.index = df_show.index + 1

                st.dataframe(df_show, use_container_width=True)

                csv = df_show.to_csv(index=True).encode("utf-8")

                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="ids_live_results.csv",
                    mime="text/csv"
                )



else:
    st.info("Upload dataset and run detection")


# ================= FOOTER =================
st.markdown("""
<div class="footer">
Intrusion Detection System · Live Detection Console · 2026
</div>
""", unsafe_allow_html=True)
