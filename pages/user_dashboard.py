import streamlit as st
from common.session import require_login, require_role

require_login()
require_role("user")

st.title("User Dashboard")

st.write("• Upload your dataset")
st.write("• View detection results")
st.write("• Search attacks")
st.write("• See top attacks")

st.info("Go to Detection Page from sidebar")
