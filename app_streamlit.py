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

# Custom CSS for Premium, Offline-First Design
st.markdown("""
<style>
    /* Google Fonts Import */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Premium Gradients */
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        background: linear-gradient(135deg, #FF4B4B 0%, #3B82F6 50%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        text-shadow: 0px 4px 20px rgba(59, 130, 246, 0.1);
    }
    
    .subtitle {
        color: #94A3B8;
        font-size: 1.15rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Sidebar styling */
    .sidebar-logo {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF4B4B 0%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
        text-align: center;
        border-bottom: 2px solid rgba(255, 255, 255, 0.05);
        padding-bottom: 0.8rem;
    }
    
    .sidebar-section {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Welcome Card */
    .welcome-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(59, 130, 246, 0.2);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    
    .welcome-header {
        font-family: 'Space Grotesk', sans-serif;
        color: #F8FAFC;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }
    
    .welcome-text {
        color: #CBD5E1;
        font-size: 1rem;
        line-height: 1.5;
    }
    
    /* Custom source tag inside chat */
    .source-header {
        color: #3B82F6;
        font-weight: 600;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        margin-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to run async RAG pipeline
async def run_local_rag(query: str, top_k: int = 2):
    from foundry_local_sdk import Configuration, FoundryLocalManager
    
    # 1. Doküman araması (Retrieval)
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

# Main Layout Headers
st.markdown("<h1 class='main-title'>🤖 Local RAG AI Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Microsoft Foundry Local & SQLite ile Güçlendirilmiş Çevrimdışı Bilgi Sistemi</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    # 100% Offline ve Şık Başlık
    st.markdown("<div class='sidebar-logo'>⚡ FOUNDRY LOCAL</div>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Ayarlar & Dosya Yönetimi")
    
    # Dosya Yükleyici (Ingestion)
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1.1rem;font-weight:700;'>📁 Yeni Doküman Yükle (.txt)</span>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Sürükle-bırak veya dosya seç", type=["txt"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        if st.button("Veritabanına İndeksle", use_container_width=True):
            os.makedirs("documents", exist_ok=True)
            file_path = os.path.join("documents", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            with st.spinner("Embedding üretiliyor..."):
                run_ingestion()
            st.success(f"'{uploaded_file.name}' indekslendi!")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Aktif Doküman Listesi
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1.1rem;font-weight:700;'>📚 Kütüphanedeki Dosyalar</span>", unsafe_allow_html=True)
    if os.path.exists("documents"):
        files = [f for f in os.listdir("documents") if f.endswith(".txt")]
        if files:
            for file in files:
                st.markdown(f"📄 `{file}`")
        else:
            st.markdown("<span style='color:#64748B;'>*Kütüphane boş.*</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span style='color:#64748B;'>*Kütüphane boş.*</span>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Model Bilgileri
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1.1rem;font-weight:700;'>🧠 Yerel Modeller</span>", unsafe_allow_html=True)
    st.markdown("🌐 **Embedding:** `qwen3-embedding-0.6b` (1024 d)")
    st.markdown("💬 **LLM:** `phi-4-mini` (3.8B Parametre)")
    st.markdown("⚡ **Hızlandırma:** CPU/GPU/NPU (Otomatik)")
    st.markdown("</div>", unsafe_allow_html=True)

# Karşılama Kartı (Sohbet geçmişi boşsa gösterilir)
if not st.session_state.messages:
    st.markdown("""
    <div class='welcome-card'>
        <div class='welcome-header'>👋 Çevrimdışı Bilgi Sistemine Hoş Geldiniz!</div>
        <div class='welcome-text'>
            Bu uygulama, yerel cihazınızdaki <b>staj rehberi</b> ve <b>Foundry Local</b> belgelerini temel alan
            özel bir yapay zeka asistanıdır. Sorduğunuz sorular için veritabanında anlamsal (semantic) arama yapılır
            ve bulunan kaynaklar <b>Phi-4-mini</b> modeline beslenerek güvenilir cevaplar üretilir.
            <br><br>
            <b>Başlamak için aşağıdaki örnek sorulardan birini yazabilirsiniz:</b>
            <ul>
                <li><i>Mentor Barbaros Bey ile haftalık toplantı ne zaman yapılacak?</i></li>
                <li><i>Final staj raporunun son teslim tarihi ve saati nedir?</i></li>
                <li><i>Microsoft Foundry Local hangi donanım hızlandırıcıları destekler?</i></li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Sohbet Geçmişini Görüntüle (Şık Avatarlar ile)
for msg in st.session_state.messages:
    avatar = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if "chunks" in msg and msg["chunks"]:
            with st.expander("🔍 Alakalı Kaynak Parçalarını Göster"):
                for chunk in msg["chunks"]:
                    st.markdown(f"<div class='source-header'>Kaynak: {chunk['title']} (Benzerlik Skoru: {chunk['score']:.4f})</div>", unsafe_allow_html=True)
                    st.info(chunk["content"])

# Kullanıcı Girişi
if user_query := st.chat_input("Yerel dokümanlarınız hakkında soru sorun..."):
    # Kullanıcı mesajını ekle
    st.chat_message("user", avatar="👤").markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Yanıt üretme
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Yerel model arama yapıyor ve cevap hazırlıyor (Çevrimdışı)..."):
            answer, chunks = asyncio.run(run_local_rag(user_query))
            
            st.markdown(answer)
            
            # Kaynak dökümanları göster
            if chunks:
                with st.expander("🔍 Alakalı Kaynak Parçalarını Göster"):
                    for chunk in chunks:
                        st.markdown(f"<div class='source-header'>Kaynak: {chunk['title']} (Benzerlik Skoru: {chunk['score']:.4f})</div>", unsafe_allow_html=True)
                        st.info(chunk["content"])
                        
            # Mesaj geçmişine ekle
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "chunks": chunks
            })
