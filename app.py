__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="IDX AI Research Engine",
    page_icon="🤖",
    layout="wide"
)

# ======================
# MODERN STYLE (CHAT AI LOOK)
# ======================
st.markdown("""
<style>
    /* Mengatur tema dasar agar konsisten di Dark Mode */
    .stApp {
        background-color: #0E1117;
        color: #E0E0E0;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }

    /* Kotak Hasil Analisis gaya Chat Assistant */
    .result-box {
        background-color: #161B22;
        color: #E0E0E0;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #30363D;
        line-height: 1.6;
        margin-top: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }

    /* Tombol Jalankan Analisis */
    .stButton > button {
        width: 100%;
        background-color: #238636;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton > button:hover {
        background-color: #2ea043;
        border: none;
    }

    /* Header Styling */
    h1 {
        font-weight: 800;
        letter-spacing: -1px;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD VECTOR DATABASE & GROQ
# ======================
@st.cache_resource
def init_models():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    db_path = "chroma_db_idx"
    vector_db = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    os.environ["GROQ_API_KEY"] = "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"
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    
    return vector_db, llm

vector_db, llm = init_models()

# ======================
# RAG LOGIC
# ======================
template = """
Anda adalah Senior Financial Analyst Profesional. 
Jawablah pertanyaan berdasarkan konteks laporan keuangan yang diberikan.

KONTEKS:
{context}

PERTANYAAN:
{question}

JAWABAN:
"""
prompt = ChatPromptTemplate.from_template(template)

def run_idx_research(ticker, query):
    # Retrieval
    results = vector_db.similarity_search(query, k=4, filter={"ticker": ticker})
    if not results:
        return None, None
        
    context_text = "\n\n".join([f"Sumber: {d.metadata['source']}\n{d.page_content}" for d in results])
    
    # Chain
    chain = ({"context": RunnablePassthrough(), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    response = chain.invoke({"context": context_text, "question": query})
    
    sources = list(set([d.metadata["source"] for d in results]))
    return response, sources

# ======================
# SIDEBAR (SETTINGS & DEPLOY)
# ======================
with st.sidebar:
    st.title("🤖 IDX AI Engine")
    st.caption("v1.2 - Advanced RAG System")
    st.divider()
    
    # Input Ticker di Sidebar agar main area bersih
    ticker = st.text_input("📍 Masukkan Ticker Emiten", value="RISE").upper()
    
    st.info("Sistem akan menganalisis dokumen berdasarkan database yang tersimpan di GitHub.")

# ======================
# MAIN INTERFACE (CHAT STYLE)
# ======================
st.title("📈 Financial Research Assistant")
st.markdown("Tanyakan apa saja mengenai laporan keuangan emiten yang terdaftar.")

# Area Input pertanyaan di bawah judul
query = st.text_area("💬 Apa yang ingin Anda ketahui?", placeholder="Contoh: Berapa laba bersih dan total aset perusahaan?", height=100)

if st.button("Jalankan Analisis"):
    if ticker and query:
        with st.spinner("Sedang menganalisis laporan keuangan..."):
            ans, docs = run_idx_research(ticker, query)
            
            if not ans:
                st.warning(f"⚠️ Data untuk ticker {ticker} tidak ditemukan.")
            else:
                st.subheader("📊 Hasil Analisis")
                # Tampilan Box hasil gaya AI Assistant
                st.markdown(f'<div class="result-box">{ans}</div>', unsafe_allow_html=True)
                
                # Menampilkan sumber dengan Expander agar lebih clean
                with st.expander("📄 Lihat Referensi Dokumen Asli"):
                    for d in docs:
                        st.write(f"- {d}")
    else:
        st.error("Harap isi Ticker dan Pertanyaan.")
