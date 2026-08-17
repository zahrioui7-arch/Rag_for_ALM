# frontend.py
import streamlit as st
import requests

st.set_page_config(page_title="ALM Knowledge Assistant", page_icon="🏦")
st.title("🏦 ALM Guidance AI")
st.markdown("Professional Asset Liability Management Support")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history from session state on every rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask me about LCR, NSFR, or Gap Analysis..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        response_placeholder = st.empty() # Create a placeholder for the stream
        full_response = ""
        
        # Call the FastAPI backend
        try:
            # We use stream=True to handle the token-by-token delivery
            with requests.post("http://localhost:8000/chat", 
                               json={"prompt": prompt}, 
                               stream=True) as r:
                for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        full_response += chunk
                        # Update the placeholder with the current text
                        # Streamlit's markdown renderer handles the interpretor part
                        response_placeholder.markdown(full_response + "▌")
                
                # Final update to remove the cursor ▌
                response_placeholder.markdown(full_response)
                
        except Exception as e:
            st.error(f"Connection Error: {e}")
            full_response = "I'm sorry, I'm having trouble connecting to the ALM server."
            response_placeholder.markdown(full_response)

    # Save assistant response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})