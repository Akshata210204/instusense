import streamlit as st
import pandas as pd
from common.session import require_login, require_role
from common.detection_utils import detect_severity

if "live_index" not in st.session_state or st.session_state["live_index"] == 0:
    st.warning("Live stream has not started yet.")
    st.stop()


# ================= AUTH =================
require_login()
require_role("user")

# ================= PAGE CONFIG =================
st.set_page_config(page_title="IDS Security Console", layout="wide")

# ================= GLOBAL CSS =================
st.markdown("""
<style>

/* ---------- HEADER ---------- */
.header {
    position: fixed;
    top: 3.5rem;
    left: 0;
    right: 0;
    height: 70px;
    background: linear-gradient(90deg, #020617, #0f172a);
    color: white;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 40px;
    z-index: 1000;
    box-shadow: 0 4px 18px rgba(0,0,0,0.3);
}

.header-title {
    font-size: 20px;
    font-weight: 600;
}

.header-user {
    font-size: 14px;
    color: #cbd5f5;
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

/* ---------- PAGE SPACING ---------- */
.block-container {
    padding-top: 140px;
    padding-bottom: 90px;
}

/* ---------- KPI CARDS ---------- */
.card {
    background: white;
    padding: 22px;
    border-radius: 16px;
    box-shadow: 0 8px 28px rgba(0,0,0,0.08);
    border-left: 6px solid #2563eb;
}

.card-title {
    font-size: 14px;
    color: #475569;
}

.card-value {
    font-size: 30px;
    font-weight: 700;
    margin-top: 8px;
    color: #020617;
}

.card.red { border-left-color: #dc2626; }
.card.orange { border-left-color: #f97316; }
.card.green { border-left-color: #16a34a; }
.card.blue { border-left-color: #2563eb; }

/* ---------- RISK PANEL ---------- */
.risk-visual {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: white;
    border-radius: 22px;
    height: 360px;
    padding: 36px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.5);
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.risk-level {
    font-size: 36px;
    font-weight: 800;
    margin: 10px 0;
}

.risk-text {
    font-size: 14px;
    color: #e5e7eb;
    line-height: 1.6;
}

/* ---------- THREAT DISTRIBUTION ---------- */
.threat-panel {
    background: #ffffff;
    border-radius: 22px;
    height: 360px;
    padding: 30px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.threat-title {
    font-size: 15px;
    color: #475569;
    margin-bottom: 20px;
}

.threat-row {
    margin-bottom: 18px;
}

.threat-label {
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    color: #334155;
    margin-bottom: 6px;
}

.threat-bar {
    width: 100%;
    height: 14px;
    background: #e5e7eb;
    border-radius: 10px;
    overflow: hidden;
}

.threat-fill {
    height: 100%;
    width: 0;
    border-radius: 10px;
    animation: fillBar 1.6s ease forwards;
}

.high { background: linear-gradient(90deg, #ef4444, #b91c1c); }
.medium { background: linear-gradient(90deg, #f59e0b, #d97706); }
.low { background: linear-gradient(90deg, #22c55e, #16a34a); }

@keyframes fillBar {
    from { width: 0; }
    to { width: var(--target); }
}

</style>
""", unsafe_allow_html=True)

# ================= LOAD DATA =================
if "last_detection" not in st.session_state:
    st.warning("No detection data found. Run detection first.")
    st.stop()

full_df = st.session_state["last_detection"]
df = full_df.iloc[:st.session_state["live_index"]].copy()

# ================= METRICS =================
total = len(df)
attack_df = df[df["Predicted_Attack"] != "Normal"]
attacks = len(attack_df)

high = (df["Severity"] == "High").sum()
medium = (df["Severity"] == "Medium").sum()
low = (df["Severity"] == "Low").sum()

ratio = round((attacks / total) * 100, 2) if total else 0

user_email = st.session_state.get("email", "user")
user_name = user_email.split("@")[0].capitalize()

# ================= HEADER =================
st.markdown(f"""
<div class="header">
<div class="header-title">IDS Security Console</div>
<div class="header-user">Welcome, {user_name}</div>
</div>
""", unsafe_allow_html=True)

# ================= KPI CARDS =================
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
<div class="card blue">
<div class="card-title">Traffic Analyzed</div>
<div class="card-value">{total:,}</div>
</div>
""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
<div class="card orange">
<div class="card-title">Threats Detected</div>
<div class="card-value">{attacks}</div>
</div>
""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
<div class="card red">
<div class="card-title">High Severity Alerts</div>
<div class="card-value">{high}</div>
</div>
""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
<div class="card green">
<div class="card-title">Attack Ratio</div>
<div class="card-value">{ratio}%</div>
</div>
""", unsafe_allow_html=True)

# ================= DISTRIBUTION + RISK =================
st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns(2)

with left:
    total_events = high + medium + low or 1
    high_pct = round((high / total_events) * 100, 1)
    med_pct = round((medium / total_events) * 100, 1)
    low_pct = round((low / total_events) * 100, 1)

    st.markdown(f"""
<div class="threat-panel">
<div class="threat-title">Threat Severity Distribution</div>

<div class="threat-row">
<div class="threat-label"><span>High</span><span>{high} ({high_pct}%)</span></div>
<div class="threat-bar"><div class="threat-fill high" style="--target:{high_pct}%;"></div></div>
</div>

<div class="threat-row">
<div class="threat-label"><span>Medium</span><span>{medium} ({med_pct}%)</span></div>
<div class="threat-bar"><div class="threat-fill medium" style="--target:{med_pct}%;"></div></div>
</div>

<div class="threat-row">
<div class="threat-label"><span>Low</span><span>{low} ({low_pct}%)</span></div>
<div class="threat-bar"><div class="threat-fill low" style="--target:{low_pct}%;"></div></div>
</div>

</div>
""", unsafe_allow_html=True)

with right:
    if high > 0:
        level = "HIGH RISK"
        desc = "Critical intrusions detected. Immediate response required."
    elif attacks > 0:
        level = "MODERATE RISK"
        desc = "Suspicious traffic patterns observed. Monitor closely."
    else:
        level = "LOW RISK"
        desc = "Network traffic is clean. No malicious activity found."

    st.markdown(f"""
<div class="risk-visual">
<div style="font-size:14px;color:#94a3b8;">Overall Risk Status</div>
<div class="risk-level">{level}</div>
<div class="risk-text">{desc}</div>
</div>
""", unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("""
<div class="footer">
Intrusion Detection System · User Security Console · 2026
</div>
""", unsafe_allow_html=True)
