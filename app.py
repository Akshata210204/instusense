import streamlit as st
from database import create_users_table
from auth import register_user, login_user

st.set_page_config(page_title="IDS Web App", layout="centered")
create_users_table()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.email = None

# ---------------- LOGIN PAGE ----------------
if not st.session_state.logged_in:
    st.title("Intrusion Detection System")

    tab1, tab2 = st.tabs(["Login", "User Register"])

    # -------- LOGIN --------
    with tab1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            user = login_user(email, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.email = email
                st.session_state.role = user[0]
                st.success("Login successful")
                st.rerun()
            else:
                st.error("Invalid credentials")

    # -------- USER REGISTER ONLY --------
    with tab2:
        r_email = st.text_input("User Email")
        r_password = st.text_input("User Password", type="password")

        if st.button("Register"):
            if register_user(r_email, r_password):
                st.success("Registered successfully. Please login.")
            else:
                st.error("Email already exists")

# ---------------- AFTER LOGIN ----------------
else:
    st.sidebar.success(f"Logged in as {st.session_state.email}")
    st.sidebar.info(f"Role: {st.session_state.role}")

    if st.sidebar.button("Logout"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.title("Welcome to IDS Web App")
    st.write("Use sidebar to navigate.")
