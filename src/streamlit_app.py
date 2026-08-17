# frontend.py
# import streamlit as st
# import requests

# st.set_page_config(page_title="ALM Knowledge Assistant", page_icon="🏦")
# st.title("🏦 ALM Guidance AI")
# st.markdown("Professional Asset Liability Management Support")

# # Initialize chat history in session state
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat history from session state on every rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # React to user input
# if prompt := st.chat_input("Ask me about LCR, NSFR, or Gap Analysis..."):
#     # Display user message in chat message container
#     st.chat_message("user").markdown(prompt)
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     # Display assistant response in chat message container
#     with st.chat_message("assistant"):
#         response_placeholder = st.empty() # Create a placeholder for the stream
#         full_response = ""
        
#         # Call the FastAPI backend
#         try:
#             # We use stream=True to handle the token-by-token delivery
#             with requests.post("http://localhost:8000/chat", 
#                                json={"prompt": prompt}, 
#                                stream=True) as r:
#                 for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
#                     if chunk:
#                         full_response += chunk
#                         # Update the placeholder with the current text
#                         # Streamlit's markdown renderer handles the interpretor part
#                         response_placeholder.markdown(full_response + "▌")
                
#                 # Final update to remove the cursor ▌
#                 response_placeholder.markdown(full_response)
                
#         except Exception as e:
#             st.error(f"Connection Error: {e}")
#             full_response = "I'm sorry, I'm having trouble connecting to the ALM server."
#             response_placeholder.markdown(full_response)

#     # Save assistant response to history
#     st.session_state.messages.append({"role": "assistant", "content": full_response})


# backend.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from generation_engine_complete import ALM_RAG_System  # Imports your core engine file

app = FastAPI()

# Initialize your master RAG system pointing to your PDF documents folder
rag = ALM_RAG_System("./alm_docs")

class QueryRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(request: QueryRequest):
    def stream_generator():
        # 1. Run pipeline steps
        standalone_query = request.prompt # (Optional: add memory condensation if needed)
        optimized_query = rag.rewriter.rewrite(standalone_query)
        context = rag.retriever.retrieve(optimized_query)
        history = "" # Optional: pass chat history if your memory layer tracks it
        
        # 2. Format prompt and call the LLM stream method
        prompt = rag.generator.prompt_template.format(
            context="\n\n".join(context), 
            history=history, 
            query=optimized_query
        )
        
        for chunk in rag.generator.llm.stream(prompt):
            yield chunk  # Streams token-by-token directly to the frontend

    return StreamingResponse(stream_generator(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)