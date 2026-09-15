import streamlit as st
import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="ALM Knowledge Assistant", page_icon="🏦")
st.title("🏦 ALM Guidance AI")
st.markdown("Professional Asset Liability Management Support")

# --- Identify the user (simple name for now; swap for real auth later) ---
if "username" not in st.session_state:
    st.session_state.username = None

if not st.session_state.username:
    with st.form("login_form"):
        name = st.text_input("Enter your name to start/continue a session")
        submitted = st.form_submit_button("Continue")
        if submitted and name.strip():
            st.session_state.username = name.strip()
            st.rerun()
    st.stop()

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def load_sessions():
    try:
        r = requests.get(f"{BACKEND_URL}/sessions", params={"username": st.session_state.username})
        r.raise_for_status()
        return r.json()
    except Exception:
        return []


def load_messages(session_id):
    try:
        r = requests.get(f"{BACKEND_URL}/sessions/{session_id}/messages")
        r.raise_for_status()
        return [{"role": m["role"], "content": m["content"]} for m in r.json()]
    except Exception:
        return []


# --- Sidebar: session list / switcher ---
with st.sidebar:
    st.markdown(f"**User:** {st.session_state.username}")
    if st.button("➕ New conversation"):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("**Past conversations**")
    for s in load_sessions():
        label = s["title"] or "Untitled"
        if st.button(label, key=s["id"]):
            st.session_state.session_id = s["id"]
            st.session_state.messages = load_messages(s["id"])
            st.rerun()

# --- Chat history ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- React to user input ---
if prompt := st.chat_input("Ask me about LCR, NSFR, or Gap Analysis..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            payload = {"prompt": prompt, "username": st.session_state.username}
            if st.session_state.session_id:
                payload["session_id"] = st.session_state.session_id

            with requests.post(f"{BACKEND_URL}/chat", json=payload, stream=True) as r:
                # Backend returns the session id (new or existing) in a header
                returned_session_id = r.headers.get("X-Session-Id")
                if returned_session_id:
                    st.session_state.session_id = returned_session_id

                for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        full_response += chunk
                        response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

        except Exception as e:
            st.error(f"Connection Error: {e}")
            full_response = "I'm sorry, I'm having trouble connecting to the ALM server."
            response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
