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
    /* Menggunakan variabel sistem Streamlit agar adaptif terhadap Light/Dark Mode */
    .result-box {
        background-color: var(--secondary-background-color);
        color: var(--text-color);
        padding: 25px;
        border-radius: 12px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        line-height: 1.7;
        margin-top: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        font-size: 1.05rem;
    }

    /* Tombol Jalankan Analisis - Modern Green */
    .stButton > button {
        width: 100%;
        background-color: #238636;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #2ea043;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    /* Mempercantik sidebar */
    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.1);
    }

    /* Styling link di footer sidebar */
    .partner-link {
        color: var(--text-color);
        text-decoration: none;
        font-weight: 500;
        opacity: 0.8;
    }
    .partner-link:hover {
        opacity: 1;
        color: #238636;
    }
</style>
""", unsafe_allow_html=True)

# ======================
# LOAD MODELS
# ======================
@st.cache_resource
def init_models():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    # Gunakan path relatif yang aman untuk Streamlit Cloud
    db_path = "chroma_db_idx" if os.path.exists("chroma_db_idx") else "RAG-Quantitative-Research-Engine-for-IDX/chroma_db_idx"
    
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
Jawablah pertanyaan berdasarkan konteks laporan keuangan yang diberikan secara mendalam dan akurat.

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
# SIDEBAR (SETTINGS & CREDITS)
# ======================
with st.sidebar:
    st.title("🤖 IDX AI Engine")
    st.caption("Advanced RAG Finance Assistant")
    st.divider()
    
    ticker = st.text_input("📍 Ticker Emiten", value="RISE", placeholder="Contoh: BBCA").upper()
    st.caption("Analisis berbasis dataset laporan keuangan 50 emiten IDX.")
    
    st.divider()
    
    # BAGIAN CREDIT PARTNER
    st.markdown("### 👥 Collaborators")
    # Link Github Naufallm
    st.markdown(f'🔗 <a href="https://github.com/Naufallm" class="partner-link">Naufallm</a>', unsafe_allow_html=True)
    # Link Github Syahrialfaturr
    st.markdown(f'🔗 <a href="https://github.com/syahrialfaturr" class="partner-link">syahrialfaturr</a>', unsafe_allow_html=True)
    
    st.divider()
    st.link_button("🚀 Deploy to Cloud", "https://share.streamlit.io/", use_container_width=True)

# ======================
# MAIN INTERFACE
# ======================
st.title("📈 Financial Research Assistant")
st.markdown("Mesin riset kuantitatif cerdas untuk ekstraksi data laporan keuangan IDX.")

query = st.text_area("💬 Apa yang ingin Anda ketahui?", placeholder="Contoh: Berapa total laba bersih dan aset lancar perusahaan?", height=120)

if st.button("Jalankan Analisis"):
    if ticker and query:
        with st.spinner(f"Sedang menganalisis data {ticker}..."):
            ans, docs = run_idx_research(ticker, query)
            
            if not ans:
                st.warning(f"⚠️ Data untuk ticker {ticker} tidak ditemukan di database.")
            else:
                st.subheader(f"📊 Hasil Analisis: {ticker}")
                st.markdown(f'<div class="result-box">{ans}</div>', unsafe_allow_html=True)
                
                with st.expander("📄 Lihat Sumber Referensi"):
                    for d in docs:
                        st.write(f"• {d}")
    else:
        st.error("Silakan isi Ticker dan Pertanyaan terlebih dahulu.")
