import streamlit as st
from common.session import require_login, require_role

require_login()
require_role("admin")

st.title("Admin Dashboard")

st.write("• Monitor system")
st.write("• Control real-time IDS")
st.write("• View global statistics")
st.write("• Model performance")

st.warning("Admin controls only")
