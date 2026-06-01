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
    page_icon="📈",
    layout="wide"
)

# ======================
# THEME-AWARE MODERN STYLE
# ======================
st.markdown("""
<style>
    .result-box {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        padding: 25px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        line-height: 1.7;
        margin-top: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
    }
    .partner-link {
        color: var(--text-color);
        text-decoration: none;
        font-weight: 500;
    }
    /* Style untuk tombol rekomendasi pertanyaan */
    .rec-button {
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD MODELS & DATA
# ======================
@st.cache_resource
def init_models():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    db_path = "chroma_db_idx" if os.path.exists("chroma_db_idx") else "RAG-Quantitative-Research-Engine-for-IDX/chroma_db_idx"
    
    vector_db = Chroma(
        persist_directory=db_path,
        embedding_function=embedding_model,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    # Ambil daftar ticker unik dari metadata database
    try:
        metadata = vector_db.get()
        tickers = sorted(list(set([m['ticker'] for m in metadata['metadatas']])))
    except:
        tickers = ["RISE", "BBCA", "IBFN"] # Fallback jika gagal

    os.environ["GROQ_API_KEY"] = "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"
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    
    return vector_db, llm, tickers

vector_db, llm, list_tickers = init_models()

# ======================
# RAG LOGIC
# ======================
template = """
Anda adalah Senior Financial Analyst Profesional. 
Jawablah pertanyaan berdasarkan konteks laporan keuangan yang diberikan secara mendalam.

KONTEKS:
{context}

PERTANYAAN:
{question}

JAWABAN:
"""
prompt = ChatPromptTemplate.from_template(template)

def run_idx_research(ticker, query):
    results = vector_db.similarity_search(query, k=4, filter={"ticker": ticker})
    if not results:
        return None, None
    context_text = "\n\n".join([f"Sumber: {d.metadata['source']}\n{d.page_content}" for d in results])
    chain = ({"context": RunnablePassthrough(), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    response = chain.invoke({"context": context_text, "question": query})
    sources = list(set([d.metadata["source"] for d in results]))
    return response, sources

# ======================
# SIDEBAR
# ======================
with st.sidebar:
    st.title("🤖 IDX AI Engine")
    st.caption("Advanced RAG Finance Assistant")
    st.divider()
    
    # DROPDOWN TICKER
    selected_ticker = st.selectbox("📍 Pilih Ticker Emiten", list_tickers, help="Daftar ini diambil otomatis dari database 50 emiten.")
    
    st.divider()
    st.markdown("### 👥 Collaborators")
    st.markdown(f'🔗 <a href="https://github.com/Naufallm" class="partner-link">Naufallm</a>', unsafe_allow_html=True)
    st.markdown(f'🔗 <a href="https://github.com/syahrialfaturr" class="partner-link">syahrialfaturr</a>', unsafe_allow_html=True)

# ======================
# MAIN INTERFACE
# ======================
st.title("📈 Financial Research Assistant")
st.markdown("Pilih emiten di sidebar dan ajukan pertanyaan mengenai performa keuangannya.")

# Fungsi untuk menangani rekomendasi pertanyaan
if 'query_input' not in st.session_state:
    st.session_state.query_input = ""

def set_query(q):
    st.session_state.query_input = q

# REKOMENDASI PERTANYAAN
st.markdown("##### 💡 Rekomendasi Pertanyaan:")
cols = st.columns(3)
with cols[0]:
    if st.button("💰 Laba Bersih?", on_click=set_query, args=("Berapa total laba bersih perusahaan pada periode ini?",)): pass
    if st.button("🏢 Total Aset?", on_click=set_query, args=("Berapa jumlah total aset perusahaan?",)): pass
with cols[1]:
    if st.button("🧾 Liabilitas?", on_click=set_query, args=("Berapa total liabilitas atau hutang perusahaan?",)): pass
    if st.button("📈 Arus Kas?", on_click=set_query, args=("Bagaimana kondisi arus kas operasional perusahaan?",)): pass
with cols[2]:
    if st.button("👤 Direksi?", on_click=set_query, args=("Siapa saja jajaran direksi atau manajemen kunci perusahaan?",)): pass
    if st.button("⚖️ Opini Auditor?", on_click=set_query, args=("Bagaimana opini auditor terhadap laporan keuangan ini?",)): pass

st.divider()

# INPUT AREA
query = st.text_area("💬 Apa yang ingin Anda ketahui?", 
                     value=st.session_state.query_input, 
                     placeholder="Tulis pertanyaan Anda di sini...", 
                     height=120)

if st.button("🚀 Jalankan Analisis"):
    if selected_ticker and query:
        with st.spinner(f"Menganalisis data {selected_ticker}..."):
            ans, docs = run_idx_research(selected_ticker, query)
            if not ans:
                st.warning("⚠️ Data tidak ditemukan.")
            else:
                st.subheader(f"📊 Hasil Analisis: {selected_ticker}")
                st.markdown(f'<div class="result-box">{ans}</div>', unsafe_allow_html=True)
                with st.expander("📄 Lihat Sumber Referensi"):
                    for d in docs:
                        st.write(f"• {d}")
    else:
        st.error("Silakan pilih ticker dan masukkan pertanyaan.")
