# app.py - Advanced AIOps & Python Incident Investigator (Enhanced UI)
import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="IT TECHNICAL Incident Investigator",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Custom CSS Injection for Modern Dark Glassmorphism Styling ---
st.markdown("""
    <style>
    /* Main Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }
    
    /* Header Container */
    .main-header {
        background: rgba(30, 41, 59, 0.7);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .main-header h1 {
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    /* Badge styling */
    .badge-container {
        display: flex;
        justify-content: center;
        gap: 12px;
        margin-top: 10px;
    }
    
    .badge {
        background: rgba(99, 102, 241, 0.2);
        color: #a5b4fc;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        border: 1px solid rgba(129, 140, 248, 0.3);
        font-weight: 600;
    }

    /* Source Evidence Cards */
    .evidence-card {
        background: rgba(30, 41, 59, 0.6);
        border-left: 4px solid #38bdf8;
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Output Report Box */
    .report-box {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(129, 140, 248, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        border-radius: 10px !important;
        padding: 0.6rem 2rem !important;
        border: none !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(168, 85, 247, 0.6) !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. Top Banner Header ---
st.markdown("""
    <div class="main-header">
        <h1>⚡ IT Technical Incident Investigator</h1>
        <p style="color: #94a3b8; font-size: 1.1rem; margin-bottom: 0;">Autonomous Root Cause Analysis & Remediation Engine</p>
        <div class="badge-container">
            <span class="badge">🔍 Hybrid Search (Dense + BM25)</span>
            <span class="badge">🎯 Cross-Encoder Reranking</span>
            <span class="badge">🤖 Local Llama 3.2</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- 4. Sidebar Controls & Metrics ---
with st.sidebar:
    st.image("https://img.icons8.com/isometric-reflection/100/server.png", width=70)
    st.title("⚙️ System Status")
    
    st.success("🟢 **Pipeline Status:** Operational")
    st.info("🧠 **Embedding Model:** `bge-small-en-v1.5`\n\n🎯 **Reranker:** `ms-marco-MiniLM-L-6-v2`\n\n🤖 **LLM Engine:** Local Ollama (`llama3.2`)")
    
    st.markdown("---")
    st.markdown("### 💡 Quick Preset Queries")
    preset_query = st.selectbox(
        "Select a sample incident/error:",
        [
            "Custom Input",
            "KeyError",
            "IndexError",
            "ERR_DATABASE_TIMEOUT",
            "ERR_MEMORY_LIMIT_EXCEEDED",
            "ERR_RATE_LIMIT_EXCEEDED"
        ]
    )
    
    st.markdown("---")
    st.caption("Built for High-Performance Enterprise Incident Diagnostics.")

# --- 5. Caching Knowledge Base & Models ---
@st.cache_resource
def initialize_rag_system():
    urls = [
        "https://docs.python.org/3/tutorial/errors.html",
        "https://docs.python.org/3/library/exceptions.html"
    ]
    web_loader = WebBaseLoader(urls)
    web_docs = web_loader.load()
    
    sre_incident_data = [
        {"alert_id": "ALT-9021", "service": "checkout-service", "error_code": "ERR_DATABASE_TIMEOUT", "message": "Connection pool exhausted while trying to reach payment gateway database."},
        {"alert_id": "ALT-9022", "service": "auth-service", "error_code": "ERR_MEMORY_LIMIT_EXCEEDED", "message": "Container killed due to OOMKilled (Out of Memory) on Kubernetes cluster node-04."},
        {"alert_id": "ALT-9023", "service": "payment-api", "error_code": "ERR_RATE_LIMIT_EXCEEDED", "message": "HTTP 429 Too Many Requests received from upstream third-party banking API."}
    ]
    
    json_docs = [Document(page_content=f"SRE Incident Alert | Alert ID: {a['alert_id']} | Service: {a['service']} | Error Code: {a['error_code']} | Log Message: {a['message']}", metadata=a) for a in sre_incident_data]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    cleaned_docs = [d for d in text_splitter.split_documents(web_docs) if not any(w in d.page_content for w in ["Table of Contents", "Previous topic", "Next topic", "Navigation"])]
    master_docs = cleaned_docs + json_docs
    
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5", model_kwargs={'device': 'cpu'})
    vectorstore = Chroma.from_documents(master_docs, embeddings, collection_name="streamlit_styled_rag")
    
    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    bm25_retriever = BM25Retriever.from_documents(master_docs)
    bm25_retriever.k = 3
    
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm = ChatOllama(model="llama3.2", temperature=0.1)
    
    return vector_retriever, bm25_retriever, reranker, llm

with st.spinner("🚀 Initializing Vector Database and Local Ollama Models..."):
    v_retriever, b_retriever, reranker_model, local_llm = initialize_rag_system()

# --- 6. Prompt Engineering ---
template = """
You are a Principal Site Reliability Engineer (SRE) and Python Core Developer. 
Analyze the retrieved context below to generate a clear, highly technical RCA report for the user's error/incident query.

--- RETRIEVED CONTEXT EVIDENCE ---
{context}
----------------------------------

USER QUERY: {query}

Provide your analysis using this markdown structure:
### 1. 🛑 Identified Exception / Alert Summary
(State what the error is and its exact domain).

### 2. 🔍 Root Cause Analysis (RCA)
(Explain technically why this error occurs based on the context).

### 3. 🛠️ Step-by-Step Remediation Plan
(Provide actionable steps, code fixes, or operational mitigation steps).

Tone: Highly professional, direct, and structured.
"""
prompt = ChatPromptTemplate.from_template(template)
rag_chain = prompt | local_llm | StrOutputParser()

# --- 7. Main Input Area ---
default_text = preset_query if preset_query != "Custom Input" else "KeyError"

col_input, col_btn = st.columns([4, 1])

with col_input:
    user_query = st.text_input("Enter Incident Log, Exception Code, or Query:", value=default_text, label_visibility="collapsed")

with col_btn:
    search_clicked = st.button("⚡ Investigate")

# --- 8. Execution and Results Presentation ---
if search_clicked or user_query:
    if user_query.strip():
        with st.spinner("🔍 Executing Hybrid Retrieval & Cross-Encoder Reranking..."):
            # Hybrid Retrieval
            v_res = v_retriever.invoke(user_query)
            b_res = b_retriever.invoke(user_query)
            
            seen, initial_docs = set(), []
            for doc in b_res + v_res:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    initial_docs.append(doc)
            
            # Cross-Encoder Rerank
            pairs = [[user_query, doc.page_content] for doc in initial_docs]
            scores = reranker_model.predict(pairs)
            scored = sorted(zip(initial_docs, scores), key=lambda x: x[1], reverse=True)
            best_docs = [doc for doc, score in scored[:2]]
            best_scores = [score for doc, score in scored[:2]]
            
            context_text = "\n\n".join([doc.page_content for doc in best_docs])

        # Two-column layout for evidence vs. LLM output
        col_evidence, col_report = st.columns([1, 1])

        with col_evidence:
            st.markdown("### 📄 Retrieved Source Evidence")
            st.markdown("*Filtered and reranked using Cross-Encoder scores*")
            
            for i, (doc, score) in enumerate(zip(best_docs, best_scores), 1):
                st.markdown(f"""
                <div class="evidence-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="color: #38bdf8; font-weight: bold;">Chunk #{i}</span>
                        <span style="background: rgba(56, 189, 248, 0.2); color: #38bdf8; padding: 2px 8px; border-radius: 12px; font-size: 0.8rem; font-weight: 600;">Relevance Score: {score:.4f}</span>
                    </div>
                    <p style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.5; margin: 0;">{doc.page_content}</p>
                </div>
                """, unsafe_allow_html=True)

        with col_report:
            st.markdown("### 🤖 SRE Incident Report")
            st.markdown("*Generated dynamically by local Llama 3.2*")
            
            with st.spinner("🤖 Synthesizing Root Cause Analysis report..."):
                ai_response = rag_chain.invoke({
                    "context": context_text,
                    "query": user_query
                })
            
            st.markdown(f"""
            <div class="report-box">
                {ai_response}
            </div>
            """, unsafe_allow_html=True)