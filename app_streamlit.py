import streamlit as st
import asyncio
import os
import json
import sqlite3
import openai
from database import get_all_documents, DB_FILE
from vector_search import VectorSearchEngine
from retrieval import retrieve_context
from ingestion import run_ingestion

# Page Configuration & Styling
st.set_page_config(
    page_title="Local RAG AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    /* Gradient Background & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        background: linear-gradient(135deg, #FF6B6B 0%, #4D96FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: #718096;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Glassmorphic Cards */
    .card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .sidebar-section {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4D96FF;
    }
    
    .source-tag {
        display: inline-block;
        background: rgba(77, 150, 255, 0.2);
        color: #4D96FF;
        border: 1px solid rgba(77, 150, 255, 0.3);
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.8rem;
        margin-right: 5px;
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to run async RAG pipeline
async def run_local_rag(query: str, top_k: int = 2):
    from foundry_local_sdk import Configuration, FoundryLocalManager
    
    # 1. Doküman araması (Retrieval)
    # qwen3-embedding modeli yüklenir, vektör aranır ve model bellekten çıkarılır
    chunks = retrieve_context(query, top_k=top_k)
    
    if not chunks:
        return "I cannot find any relevant documents in the database.", []
        
    context_text = ""
    for idx, chunk in enumerate(chunks, 1):
        context_text += f"\n[Doküman {idx}] (Kaynak: {chunk['title']})\n{chunk['content']}\n"
        
    # 2. Local LLM (phi-4-mini) başlatma ve çıkarım
    web_config = Configuration.WebService(urls="http://127.0.0.1:0")
    config = Configuration(app_name="foundry-local-test", web=web_config)
    
    try:
        FoundryLocalManager.initialize(config)
    except Exception:
        pass
        
    manager = FoundryLocalManager.instance
    catalog = manager.catalog
    
    manager.start_web_service()
    endpoint = manager.urls[0]
    
    model = catalog.get_model("phi-4-mini")
    model.load()
    
    client = openai.OpenAI(
        base_url=f"{endpoint}/v1" if not endpoint.endswith("/v1") else endpoint,
        api_key="not-needed"
    )
    
    system_prompt = (
        "You are a helpful local assistant. Answer the user's question using ONLY the provided Context below. "
        "If the Context does not contain the answer, reply with 'I cannot find this information in my database.' "
        "Do not make up facts or use external knowledge. Always keep your response grounded."
    )
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"
    
    response = client.chat.completions.create(
        model=model.id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0
    )
    
    answer = response.choices[0].message.content
    
    # Kapatma temizliği
    model.unload()
    manager.stop_web_service()
    
    return answer, chunks

# Streamlit Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

# Layout
st.markdown("<h1 class='main-title'>🤖 Local RAG AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Microsoft Foundry Local ve SQLite ile Çevrimdışı Bilgi Sistemi</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://www.foundrylocal.ai/logos/foundry-local-logo-color.svg", width=200)
    st.markdown("### ⚙️ Ayarlar & Dosya Yönetimi")
    
    # Dosya Yükleyici (Ingestion)
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("#### 📁 Yeni Doküman Yükle (.txt)")
    uploaded_file = st.file_uploader("Metin dosyası seçin", type=["txt"])
    
    if uploaded_file is not None:
        if st.button("Veritabanına İndeksle"):
            # Dosyayı documents dizinine kaydet
            os.makedirs("documents", exist_ok=True)
            file_path = os.path.join("documents", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Ingestion işlemini çalıştır
            with st.spinner("Embedding üretiliyor ve veritabanı güncelleniyor..."):
                run_ingestion()
            st.success(f"'{uploaded_file.name}' başarıyla indekslendi!")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Aktif Doküman Listesi
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("#### 📚 Kütüphanedeki Dosyalar")
    if os.path.exists("documents"):
        files = [f for f in os.listdir("documents") if f.endswith(".txt")]
        if files:
            for file in files:
                st.markdown(f"📄 `{file}`")
        else:
            st.markdown("*Kütüphane boş.*")
    else:
        st.markdown("*Kütüphane boş.*")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Model Bilgileri
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("#### 🧠 Yerel Modeller")
    st.markdown("- **Embedding**: `qwen3-embedding-0.6b` (1024 d)")
    st.markdown("- **LLM**: `phi-4-mini` (3.8B parameters)")
    st.markdown("- **Hızlandırma**: CPU/GPU/NPU (Otomatik)")
    st.markdown("</div>", unsafe_allow_html=True)

# Main Chat Interface
# Sohbet Geçmişini Görüntüle
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Kaynakları göster
        if "chunks" in msg and msg["chunks"]:
            with st.expander("📚 Alakalı Kaynakları Göster"):
                for chunk in msg["chunks"]:
                    st.markdown(f"**Kaynak:** `{chunk['title']}` (Skor: {chunk['score']:.4f})")
                    st.info(chunk["content"])

# Kullanıcı Girişi
if user_query := st.chat_input("Yerel dokümanlarınız hakkında soru sorun..."):
    # Kullanıcı mesajını ekle
    st.chat_message("user").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Yanıt üretme
    with st.chat_message("assistant"):
        with st.spinner("Yerel model cevap üretiyor (Çevrimdışı)..."):
            # Async fonksiyonu senkron olarak streamlit içinde çalıştır
            answer, chunks = asyncio.run(run_local_rag(user_query))
            
            st.markdown(answer)
            
            # Kaynak dökümanları göster
            if chunks:
                with st.expander("📚 Alakalı Kaynakları Göster"):
                    for chunk in chunks:
                        st.markdown(f"**Kaynak:** `{chunk['title']}` (Skor: {chunk['score']:.4f})")
                        st.info(chunk["content"])
                        
            # Mesaj geçmişine ekle
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "chunks": chunks
            })
