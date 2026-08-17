# 🏦 ALM Knowledge Assistant (RAG)

An enterprise-grade Retrieval-Augmented Generation (RAG) assistant designed for Asset Liability Management (ALM), powered by FastAPI, Streamlit, LangChain, and local LLMs via Ollama.

## 🚀 Features
- **Prompt Guardrails & Rewriting:** Ensures queries remain within ALM financial contexts and optimizes them for vector matching.
- **Hybrid Search:** Combines semantic ANN search (Chroma + Nomic embeddings) with keyword search (BM25).
- **Advanced Re-ranking:** Uses FlashRank to re-order the top candidate chunks for high precision.
- **Token-by-token Streaming:** Integrated FastAPI backend streaming responses to a Streamlit chat UI.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/alm-rag-assistant.git](https://github.com/YOUR_USERNAME/alm-rag-assistant.git)
   cd alm-rag-assistant