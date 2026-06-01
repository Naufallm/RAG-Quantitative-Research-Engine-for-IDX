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
# CUSTOM CSS (MODERN & ADAPTIVE)
# ======================
st.markdown("""
<style>
    /* Adaptive Result Box */
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
    
    /* Modern Green Button */
    .stButton > button {
        width: 100%;
        background-color: #238636;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px;
        font-weight: 600;
    }

    /* Collaborator Styling */
    .dev-card {
        text-align: center;
        padding: 10px;
    }
    .dev-name {
        font-size: 14px;
        font-weight: 600;
        margin-top: 5px;
    }

    /* Custom Sticky Footer */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: var(--background-color);
        text-align: center;
        padding: 10px;
        font-size: 13px;
        border-top: 1px solid rgba(128, 128, 128, 0.1);
        z-index: 100;
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
    
    # Auto-discovery Tickers
    try:
        metadata = vector_db.get()
        tickers = sorted(list(set([m['ticker'] for m in metadata['metadatas']])))
    except:
        tickers = ["RISE", "BBCA", "IBFN"]

    os.environ["GROQ_API_KEY"] = "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"
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    
    return vector_db, llm, tickers

vector_db, llm, list_tickers = init_models()

# ======================
# RAG LOGIC
# ======================
template = """
Anda adalah Senior Financial Analyst Profesional. 
Jawablah pertanyaan berdasarkan konteks laporan keuangan yang diberikan secara akurat.

KONTEKS:
{context}

PERTANYAAN:
{question}

JAWABAN:
"""
prompt = ChatPromptTemplate.from_template(template)

def run_idx_research(ticker, query):
    results = vector_db.similarity_search(query, k=4, filter={"ticker": ticker})
    if not results: return None, None
    context_text = "\n\n".join([f"Sumber: {d.metadata['source']}\n{d.page_content}" for d in results])
    chain = ({"context": RunnablePassthrough(), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    response = chain.invoke({"context": context_text, "question": query})
    sources = list(set([d.metadata["source"] for d in results]))
    return response, sources

# ======================
# SIDEBAR (SETTINGS & COLLABORATORS)
# ======================
with st.sidebar:
    st.image("https://www.idx.co.id/id/media/1346/logo-idx.png", width=80)
    st.title("🤖 IDX AI Engine")
    st.caption("Advanced RAG Finance Assistant")
    st.divider()
    
    selected_ticker = st.selectbox("📍 Pilih Ticker Emiten", list_tickers)
    
    st.divider()
    
    # COLLABORATORS SECTION WITH IMAGES
    st.markdown("### 👥 Developed By")
    col_a, col_b = st.columns(2)
    with col_a:
        st.image("https://github.com/Naufallm.png", width=70)
        st.markdown(f'<div class="dev-name"><a href="https://github.com/Naufallm" style="text-decoration:none; color:inherit;">Naufallm</a></div>', unsafe_allow_html=True)
    with col_b:
        st.image("https://github.com/syahrialfaturr.png", width=70)
        st.markdown(f'<div class="dev-name"><a href="https://github.com/syahrialfaturr" style="text-decoration:none; color:inherit;">Syahrialfaturr</a></div>', unsafe_allow_html=True)

# ======================
# MAIN INTERFACE
# ======================
st.title("📈 Financial Research Assistant")
st.markdown("Mesin riset kuantitatif untuk analisis laporan keuangan IDX secara instan.")

# Session state for recommended questions
if 'query_input' not in st.session_state:
    st.session_state.query_input = ""

def set_query(q):
    st.session_state.query_input = q

# RECOMMENDED QUESTIONS GRID
st.markdown("##### 💡 Rekomendasi Pertanyaan:")
r1, r2, r3 = st.columns(3)
with r1:
    if st.button("💰 Laba Bersih?", on_click=set_query, args=("Berapa total laba bersih perusahaan?",)): pass
    if st.button("🏢 Total Aset?", on_click=set_query, args=("Berapa jumlah total aset perusahaan?",)): pass
with r2:
    if st.button("🧾 Liabilitas?", on_click=set_query, args=("Berapa total liabilitas atau hutang perusahaan?",)): pass
    if st.button("📈 Arus Kas?", on_click=set_query, args=("Bagaimana kondisi arus kas operasional?",)): pass
with r3:
    if st.button("👤 Direksi?", on_click=set_query, args=("Siapa saja jajaran manajemen kunci perusahaan?",)): pass
    if st.button("⚖️ Opini Auditor?", on_click=set_query, args=("Bagaimana opini auditor terhadap laporan ini?",)): pass

st.divider()

# INPUT AREA
query = st.text_area("💬 Apa yang ingin Anda ketahui?", 
                     value=st.session_state.query_input, 
                     placeholder="Contoh: Berapa laba per saham perusahaan?", 
                     height=100)

if st.button("🚀 Jalankan Analisis"):
    if selected_ticker and query:
        with st.spinner(f"Menganalisis {selected_ticker}..."):
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
        st.error("Pilih ticker dan masukkan pertanyaan.")

# ======================
# CUSTOM STICKY FOOTER
# ======================
st.markdown(
    f"""
    <div class="footer">
        Built with ❤️ by 
        <a href="https://github.com/Naufallm" target="_blank" style="color: #238636; text-decoration: none; font-weight: bold;">Naufallm</a> & 
        <a href="https://github.com/syahrialfaturr" target="_blank" style="color: #238636; text-decoration: none; font-weight: bold;">syahrialfaturr</a> 
        | © 2024 IDX AI Research Engine
    </div>
    """,
    unsafe_allow_html=True
)
