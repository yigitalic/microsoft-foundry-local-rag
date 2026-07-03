import streamlit as st
import asyncio
import os
import json
import sqlite3
import openai
import pandas as pd
from database import get_all_documents, DB_FILE
from vector_search import VectorSearchEngine
from retrieval import retrieve_context
from ingestion import run_ingestion

# Page Configuration & Styling
st.set_page_config(
    page_title="Local RAG AI Assistant V2",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium, Clean Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Clean solid off-white header */
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        color: #F8FAFC;
        font-size: 3.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #94A3B8;
        font-size: 1.15rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Sidebar header styling */
    .sidebar-logo {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #3B82F6;
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
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
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
    
    /* Source block style */
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
async def run_local_rag(query: str, top_k: int = 2, file_type_filter: str = None):
    from foundry_local_sdk import Configuration, FoundryLocalManager
    
    # 1. Doküman araması (Retrieval - Hibrit Arama & Filtreleme)
    filter_val = None if file_type_filter == "Hepsi" else file_type_filter
    chunks = retrieve_context(query, top_k=top_k, file_type_filter=filter_val)
    
    if not chunks:
        return "I cannot find any relevant documents matching the query/filter in the database.", []
        
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
        "You are a highly analytical on-device AI agent. Your task is to answer the user's question based ONLY on the provided Context.\n\n"
        "Follow this multi-step reasoning protocol:\n"
        "1. Assess the sufficiency of the Context: Check if the retrieved context actually contains the specific information needed to answer the question.\n"
        "2. If the context is sufficient, provide a precise, detailed, and directly grounded answer.\n"
        "3. If the context is insufficient, explain exactly why the retrieved chunks (mentioning their sources) are not enough to answer the question, and decline to answer (e.g. 'Retrieved sources discuss X, but do not contain information to answer Y.').\n"
        "Do NOT make up facts. Never use external knowledge. Always keep your response grounded."
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
st.markdown("<h1 class='main-title'>🤖 Local RAG AI Assistant V2</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Microsoft Foundry Local & SQLite ile Güçlendirilmiş Çoklu Format Destekli Bilgi Sistemi</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<div class='sidebar-logo'>⚡ FOUNDRY LOCAL V2</div>", unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Ayarlar & Dosya Yönetimi")
    
    # Arama Filtrelemesi (Metadata Filtering)
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1.1rem;font-weight:700;'>🔍 Arama Filtresi (Metaveri)</span>", unsafe_allow_html=True)
    filter_options = ["Hepsi", "text", "pdf", "docx", "python", "javascript", "markdown"]
    selected_filter = st.selectbox(
        "Sadece şu dosya türlerinde ara:",
        options=filter_options,
        index=0,
        label_visibility="collapsed"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Dosya Yükleyici (Ingestion)
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1.1rem;font-weight:700;'>📁 Yeni Doküman / Kod Yükle</span>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Desteklenenler: .txt, .pdf, .docx, .py, .js, .md",
        type=["txt", "pdf", "docx", "py", "js", "md"],
        label_visibility="collapsed"
    )
    
    if uploaded_file is not None:
        if st.button("Veritabanına İndeksle", use_container_width=True):
            os.makedirs("documents", exist_ok=True)
            file_path = os.path.join("documents", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            with st.spinner("Embedding üretiliyor ve akıllı parçalama yapılıyor..."):
                run_ingestion()
            st.success(f"'{uploaded_file.name}' indekslendi!")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Model Bilgileri
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1.1rem;font-weight:700;'>🧠 Aktif Yerel Modeller</span>", unsafe_allow_html=True)
    st.markdown("🌐 **Embedding:** `qwen3-embedding-0.6b` (1024 d)")
    st.markdown("💬 **LLM:** `phi-4-mini` (3.8B Parametre)")
    st.markdown("🔍 **Arama:** Hibrit (Kelime + Vektör + RRF)")
    st.markdown("</div>", unsafe_allow_html=True)

# Main Application Tabs (Chat & Analytics)
tab_chat, tab_analytics = st.tabs(["💬 Sohbet Asistanı", "📊 Veri Analitiği (Dashboard)"])

# TAB 1: Chat Assistant
with tab_chat:
    # Karşılama Kartı
    if not st.session_state.messages:
        st.markdown("""
        <div class='welcome-card'>
            <div class='welcome-header'>👋 Gelişmiş Çevrimdışı Bilgi Sistemine Hoş Geldiniz (V2)!</div>
            <div class='welcome-text'>
                V2 sürümünde artık <b>PDF</b> belgeleriniz, <b>Word (Docx)</b> dokümanlarınız ve <b>Python/JavaScript</b> kod dosyalarınız
                akıllı sözdizimi algılayıcılarla (syntax-aware chunking) otomatik olarak indekslenebilmektedir.
                Kesin kelime aramalarını yakalamak için <b>TF-IDF</b> ve <b>Vektör Araması</b> birleştirilmiştir (Hybrid Search).
                <br><br>
                <b>Başlamak için bir soru yazabilir veya sol menüden filtreleme yapabilirsiniz.</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Sohbet Geçmişini Görüntüle
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if "chunks" in msg and msg["chunks"]:
                with st.expander("🔍 Alakalı Kaynak Parçalarını Göster"):
                    for chunk in msg["chunks"]:
                        st.markdown(f"<div class='source-header'>Kaynak: {chunk['title']} (RRF Skoru: {chunk['score']:.4f}, Tür: {chunk.get('file_type', 'txt').upper()})</div>", unsafe_allow_html=True)
                        st.info(chunk["content"])

    # Kullanıcı Girişi
    if user_query := st.chat_input("Yerel belgeleriniz ve kodlarınız hakkında soru sorun..."):
        st.chat_message("user", avatar="👤").markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Yerel model arama yapıyor ve cevap hazırlıyor (Çevrimdışı)..."):
                answer, chunks = asyncio.run(run_local_rag(user_query, file_type_filter=selected_filter))
                
                st.markdown(answer)
                
                # Kaynakları Göster
                if chunks:
                    with st.expander("🔍 Alakalı Kaynak Parçalarını Göster"):
                        for chunk in chunks:
                            st.markdown(f"<div class='source-header'>Kaynak: {chunk['title']} (RRF Skoru: {chunk['score']:.4f}, Tür: {chunk.get('file_type', 'txt').upper()})</div>", unsafe_allow_html=True)
                            st.info(chunk["content"])
                            
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "chunks": chunks
                })

# TAB 2: Analytics Dashboard
with tab_analytics:
    st.markdown("### 📊 Veritabanı ve Kütüphane İstatistikleri")
    
    # Verileri oku
    db_docs = get_all_documents()
    
    if db_docs:
        # DataFrame oluştur
        df = pd.DataFrame(db_docs)
        
        # Benzersiz orijinal belge sayısı (source_file kolonundan)
        unique_files = df["source_file"].nunique()
        total_chunks = len(df)
        
        # Metrikler
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Toplam İndekslenen Belge (Dosya)", unique_files)
        with col2:
            st.metric("Toplam Metin Parçası (Chunk)", total_chunks)
        with col3:
            st.metric("Saklanan Toplam Vektör", total_chunks)
            
        # Dosya Türlerine Göre Dağılım Grafiği
        st.markdown("#### 📈 Dosya Formatı Dağılımı")
        type_counts = df["file_type"].value_counts().reset_index()
        type_counts.columns = ["Dosya Türü", "Parça Sayısı"]
        
        # Grafik için Streamlit yerel bar_chart kullanımı
        chart_data = df["file_type"].value_counts()
        st.bar_chart(chart_data)
        
        # Doküman Listesi Tablosu
        st.markdown("#### 📂 İndekslenen Dosya Listesi ve Detaylar")
        doc_details = df[["source_file", "file_type", "title", "content"]].copy()
        doc_details.columns = ["Dosya Adı", "Tür", "Parça Başlığı", "İçerik Örneği"]
        # İçeriği kısaltarak gösterelim
        doc_details["İçerik Örneği"] = doc_details["İçerik Örneği"].apply(lambda x: x[:100] + "..." if len(x) > 100 else x)
        st.dataframe(doc_details, use_container_width=True)
        
    else:
        st.warning("Veritabanı boş. Lütfen sol menüden dosya yükleyin veya ingestion.py çalıştırın.")
