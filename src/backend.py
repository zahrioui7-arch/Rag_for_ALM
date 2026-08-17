import os
import re
# import numpy as np
# import chromadb
from langchain_chroma import Chroma
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from flashrank import Ranker
from langchain_community.llms import Ollama
from langchain_ollama import OllamaEmbeddings
# from langchain_community.embeddings import OllamaEmbeddings

# from langchain_community.vectorstores import Chroma
# from langchain.prompts import PromptTemplate
# from langchain.chains import LLMChain
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from flashrank import Ranker, RerankRequest
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import ollama
# ==========================================
# MODULE 1: Guardrail & Rewriter
# ==========================================
class PromptGuardrail:
    def __init__(self):
        self.allowed_keywords = [
            "alm", "liquidity", "interest rate", "gap analysis", 
            "duration", "balance sheet", "asset", "liability", 
            "risk", "regulatory", "lcr", "nsfr", "hedging", "capital"
        ]

    def validate(self, prompt: str) -> bool:
        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in self.allowed_keywords)


class PromptRewriter:
    def __init__(self, model_name="qwen3.6:35b-a3b"): 
        # Ensure the model name matches your ollama list
        self.llm = OllamaLLM(model=model_name, temperature=0.1) # Low temperature for consistency
        
        # Nemotron Nano benefits from very structured, directive prompts
        self.template = """### Instruction:
You are a professional ALM (Asset Liability Management) assistant. 
Rewrite the following user query to be a clear, formal, and detailed search query for a technical knowledge base.

### Rules:
1. Output ONLY the rewritten query.
2. Do not provide an answer.
3. Do not say "Here is the rewritten query:".
4. Use professional banking and risk management terminology.

### Original Query: 
{user_query}

### Optimized Query:"""
        
        self.prompt_template = PromptTemplate(
            input_variables=["user_query"], 
            template=self.template
        )
        self.chain = self.prompt_template | self.llm

    def rewrite(self, prompt: str) -> str:
        """
        Transforms the prompt into a version that will yield 
        better Cosine Similarity results in the Vector DB.
        """
        # We use .invoke() as .run() is being deprecated in newer LangChain versions
        response = self.chain.invoke({"user_query": prompt})
        print(type(response))

        return response.strip()

class PromptRewriter:
    def __init__(self, model_name="qwen3.6:35b-a3b"):
        self.llm = OllamaLLM(model=model_name, temperature=0.1)
        self.template = """### Instruction:
Rewrite the following user query to be a clear, formal search query for an ALM knowledge base.
Output ONLY the rewritten query. No conversation.

### Original Query: 
{user_query}

### Optimized Query:"""
        self.prompt_template = PromptTemplate(input_variables=["user_query"], template=self.template)
        self.chain = self.prompt_template | self.llm

    def rewrite(self, prompt: str) -> str:
        response = self.chain.invoke({"user_query": prompt})
        return response.strip()
# PromptRewriter().rewrite("what is the bar limit for LCR ?")

# ==========================================
# MODULE 2: PDF Loading & Cross-Chunking
# ==========================================
class ALMDocumentLoader:
    def __init__(self, folder_path):
        self.folder_path = folder_path

    def load_all_pdfs(self):
        combined_text = ""
        if not os.path.exists(self.folder_path):
            return ""
        pdf_files = [f for f in os.listdir(self.folder_path) if f.endswith('.pdf')]
        for file_name in pdf_files:
            try:
                reader = PdfReader(os.path.join(self.folder_path, file_name))
                text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
                combined_text += f"\n--- SOURCE: {file_name} ---\n{text}\n"
            except Exception as e:
                print(f"Error reading {file_name}: {e}")
        return combined_text

class ALMCrossChunker:
    def __init__(self, chunk_size=500, overlap_words=50, overlap_phrases=3):
        self.chunk_size = chunk_size
        self.overlap_words = overlap_words
        self.overlap_phrases = overlap_phrases

    def chunk_text(self, text):
        phrases = re.split(r'(?<=[.!?;])\s+', text)
        chunks = []
        current_chunk_phrases = []
        current_word_count = 0
        i = 0
        while i < len(phrases):
            phrase = phrases[i]
            current_chunk_phrases.append(phrase)
            current_word_count += len(phrase.split())
            i += 1
            if current_word_count >= self.chunk_size:
                chunks.append(" ".join(current_chunk_phrases))
                # Overlap logic
                overlap_start = max(0, i - self.overlap_phrases)
                current_chunk_phrases = phrases[overlap_start:i]
                current_word_count = sum(len(p.split()) for p in current_chunk_phrases)
                while current_word_count < self.overlap_words and overlap_start > 0:
                    overlap_start -= 1
                    current_chunk_phrases.insert(0, phrases[overlap_start])
                    current_word_count += len(phrases[overlap_start].split())
        if current_chunk_phrases:
            chunks.append(" ".join(current_chunk_phrases))
        return chunks

# ==========================================
# MODULE 3: Embedding & Vector Store (ANN)
# ==========================================
class ALMVectorStore:
    def __init__(self, model_name="nomic-embed-text"):
        # Initialize native Chroma persistent client
        self.client = chromadb.PersistentClient(path="./alm_chroma_db")
        
        # Bind the Ollama embedding function directly to the collection
        self.collection = self.client.get_or_create_collection(
            name="alm_kb_native", 
            embedding_function=OllamaEmbeddingFunction(model_name)
        )

    def add_documents(self, chunks):
        # Chroma natively takes documents, IDs, and metadata
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        self.collection.add(
            documents=chunks,
            ids=ids
        )



# 1. Create a custom wrapper for Ollama to fit Chroma's expected format
class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name="nomic-embed-text"):
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        # Ollama can take a list of texts and return a list of embeddings
        response = ollama.embed(model=self.model_name, input=list(input))
        return response['embeddings']


# ==========================================
# MODULE 4: Hybrid Retriever (Native Chroma + BM25 + Re-rank)
# ==========================================
class HybridRetriever:
    def __init__(self, vector_store, chunks):
        self.vector_store = vector_store
        self.chunks = chunks
        self.bm25 = BM25Okapi([doc.lower().split(" ") for doc in chunks])
        self.ranker = Ranker()

    def retrieve(self, query, top_k=5, candidate_pool=20):
        # ANN Search using Chroma's native query (now powered by nomic-embed-text)
        ann_results = self.vector_store.collection.query(
            query_texts=[query], 
            n_results=candidate_pool, 
            include=['documents', 'distances']
        )
        
        ann_docs = ann_results['documents'][0]
        ann_distances = ann_results['distances'][0]
        
        # Convert L2 distances to similarity scores (normalized approximation)
        ann_scores = [1.0 / (1.0 + d) for d in ann_distances]
        ann_map = {doc: score for doc, score in zip(ann_docs, ann_scores)}
        print("\n calculating semantic values finished \n")
        
        # BM25 Search
        bm25_scores = self.bm25.get_scores(query.lower().split(" "))
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        norm_bm25_scores = [s / max_bm25 for s in bm25_scores]

        print("\n calculating BM25 finished \n")
        
        # Weighted Fusion (75% ANN, 25% BM25)
        final_candidates = []
        for idx, chunk in enumerate(self.chunks):
            score = (0.75 * ann_map.get(chunk, 0)) + (0.25 * norm_bm25_scores[idx])
            final_candidates.append({"text": chunk, "score": score})

        print("\n reranking candidates finished \n")
        final_candidates = sorted(final_candidates, key=lambda x: x['score'], reverse=True)[:candidate_pool]
        
        # Re-ranking
        ranker_input = [{"id": i, "text": c['text']} for i, c in enumerate(final_candidates)]
        rerank_request = RerankRequest(query=query, passages=final_candidates)
        reranked = self.ranker.rerank(rerank_request)
        counter = 0
        for res in reranked[:top_k]:
            print("\n", counter, "\n", res['text'] )
            counter += 1
        return [res['text'] for res in reranked[:top_k]]
    print("sidfn")


# class ALMVectorStore:
#     def __init__(self, model_name="nomic-embed-text"):
#         self.embeddings = OllamaEmbeddings(model=model_name)
#         self.vector_db = Chroma(
#             collection_name="alm_kb_v2", 
#             embedding_function=self.embeddings, 
#             persist_directory="./alm_chroma_db"
#         )

#     def add_documents(self, chunks):
#         self.vector_db.add_texts(texts=chunks)

# ==========================================
# MODULE 4: Hybrid Retriever (ANN + BM25 + Re-rank)
# ==========================================
# class HybridRetriever:
    # def __init__(self, vector_store, chunks):
    #     self.vector_store = vector_store
    #     self.chunks = chunks
    #     self.bm25 = BM25Okapi([doc.lower().split(" ") for doc in chunks])
    #     self.ranker = Ranker()

    # def retrieve(self, query, top_k=3, candidate_pool=20):
    #     # ANN Search
    #     print("\n ANN searching \n")
    #     # ann_results = self.vector_store.vector_db._collection.query(
    #     #     query_texts=[query], n_results=candidate_pool, include=['documents', 'distances']
    #     # )
    #     print("\n preparing ANN results finished \n")
    #     docs_and_scores = self.vector_store.vector_db.similarity_search_with_score(query, k=candidate_pool)
        
    #     ann_map = {}
    #     for doc, distance in docs_and_scores:
    #         # Chroma returns L2 distance by default; convert distance to similarity score
    #         score = 1.0 / (1.0 + distance)
    #         ann_map[doc.page_content] = score
    #     print("\n calculating simantic values finished \n")
        
    #     # BM25 Search
    #     bm25_scores = self.bm25.get_scores(query.lower().split(" "))
    #     max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    #     norm_bm25_scores = [s / max_bm25 for s in bm25_scores]

    #     print("\n calculating BM25 finished \n")
    #     # Weighted Fusion (75% ANN, 25% BM25)
    #     final_candidates = []
    #     for idx, chunk in enumerate(self.chunks):
    #         score = (0.75 * ann_map.get(chunk, 0)) + (0.25 * norm_bm25_scores[idx])
    #         final_candidates.append({"text": chunk, "score": score})

    #     print("\n reranking candidates finished \n")
    #     final_candidates = sorted(final_candidates, key=lambda x: x['score'], reverse=True)[:candidate_pool]
        
    #     # Re-ranking
    #     ranker_input = [{"id": i, "text": c['text']} for i, c in enumerate(final_candidates)]
    #     reranked = self.ranker.rerank(query=query, docs=ranker_input)
    #     return [res['text'] for res in reranked[:top_k]]

    # print("sidfn")
print("sidfn")
# ==========================================
# MODULE 5: Generator
# ==========================================
class ALMGenerator:
    def __init__(self, model_name="qwen3.6:35b-a3b"):
        self.llm = OllamaLLM(model=model_name, temperature=0.1)
        self.template = """Instruction:
You are an ALM Specialist. Answer using ONLY the context below. 
If not in context, say "Information not available"\n
orgnize the next context to answer the query
Technical Context:
{context}
\n
User Query:
{query}
\n
Professional Guidance Answer:"""
        self.prompt_template = PromptTemplate(input_variables=["context", "query"], template=self.template)
        
    def generate_answer(self, query, retrieved_context):
        context_block = "\n\n".join(retrieved_context)
        final_prompt = self.prompt_template.format(context=context_block, query=query)
        return self.llm.invoke(final_prompt)

# test_alm_gemerator = ALMGenerator("nemotron-3-nano:4b")
# test_alm_gemerator.generate_answer("what is LCR ?", )

# ==========================================
# MASTER PIPELINE
# ==========================================
class ALM_RAG_System:   
    def __init__(self, pdf_folder):
        print("🚀 Initializing ALM RAG System...")
        
        # 1. Load PDFs
        loader = ALMDocumentLoader(pdf_folder)
        raw_text = loader.load_all_pdfs()
        if not raw_text: raise ValueError("No PDF text found!")

        print("Chunking")
        # 2. Chunking
        self.chunker = ALMCrossChunker()
        self.chunks = self.chunker.chunk_text(raw_text)

        print("Embeddings")
        # 3. Embeddings (ANN)
        self.store = ALMVectorStore()
        self.store.add_documents(self.chunks)

        print("Retrieval Components")
        # 4. Retrieval Components
        self.guard = PromptGuardrail()
        self.rewriter = PromptRewriter()
        self.retriever = HybridRetriever(self.store, self.chunks)
        
        print("Generator")
        # 5. Generator
        self.generator = ALMGenerator()
        print(f"✅ System Ready. {len(self.chunks)} chunks indexed.")

    def ask(self, user_query):
        if not self.guard.validate(user_query):
            return "❌ Error: Query outside ALM scope."
        print("paste validate !")
        optimized_query = self.rewriter.rewrite(user_query)
        print("paste validate !")
        context = self.retriever.retrieve(optimized_query)
        print("paste validate !")
        return self.generator.generate_answer(optimized_query, context)

# ==========================================
# RUN
# ==========================================
# import os
# PDF_FOLDER = "./alm_docs" 
# os.makedirs(PDF_FOLDER, exist_ok=True) 
# rag = ALM_RAG_System(PDF_FOLDER)
# user_q = "What are the liquidity GAP ?"
# print(f"\nUser: {user_q}\nAnswer: {rag.ask(user_q)}")
# optimized_query = rag.rewriter.rewrite(user_q)
# context = rag.retriever.retrieve(optimized_query)
# rag.generator.generate_answer(user_q, context)

# generator = ALMGenerator()
# answer_difinitive = generator.generate_answer(user_q, context)
# answer_difinitive



if __name__ == "__main__":
    # Folder containing your PDF files
    PDF_FOLDER = "./alm_docs" 
    os.makedirs(PDF_FOLDER, exist_ok=True) 

    try:
        rag = ALM_RAG_System(PDF_FOLDER)
        user_q = "What is the LCR ?"
        answer = rag.ask(user_q)
        print(f"\nUser: {user_q}\nAnswer: \n \n {answer}")
    except Exception as e:
        print(f"Error: {e}")