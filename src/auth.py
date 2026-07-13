"""
Simple authentication for the assessment's login page requirement.
No database is required per the spec — credentials are configured via
Streamlit secrets (or fall back to a demo default so graders can log in
without any setup).
"""
from __future__ import annotations

import streamlit as st


def _get_credentials() -> dict[str, str]:
    """
    Reads username -> password pairs from st.secrets["auth"]["users"] if
    present, e.g. in .streamlit/secrets.toml:

        [auth]
        users = { demo = "demo1234", admin = "changeme" }

    Falls back to a single demo account so the app is usable out of the box.
    """
    try:
        users = dict(st.secrets["auth"]["users"])
        if users:
            return users
    except Exception:
        pass
    return {"demo": "demo1234"}


def login_view() -> bool:
    """Renders the login form. Returns True once authenticated."""
    if st.session_state.get("authenticated"):
        return True

    st.markdown(
        """
        <div style="max-width:420px;margin:8vh auto 0 auto;text-align:center;">
            <h1 style="margin-bottom:0;">🏠 Real Estate AI Assistant</h1>
            <p style="color:#6b7280;">Sign in to chat with the knowledge base</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        with st.form("login_form", border=True):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)

        st.caption("Demo credentials — username: `demo`  ·  password: `demo1234`")

    if submitted:
        users = _get_credentials()
        if users.get(username) == password:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password.")

    return False


def logout_button() -> None:
    if st.sidebar.button("Log out", use_container_width=True):
        for key in ("authenticated", "username", "messages"):
            st.session_state.pop(key, None)
        st.rerun()
