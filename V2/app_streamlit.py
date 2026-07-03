import streamlit as st
import asyncio
import os
import json
import sqlite3
import openai
import pandas as pd
from datetime import datetime
from database import (
    init_db, get_all_documents, DB_FILE,
    create_session, get_sessions, save_message, get_session_messages, delete_session
)
from vector_search import VectorSearchEngine
from retrieval import retrieve_context
from ingestion import run_ingestion

# Veritabanını ilklendir
init_db()

# Page Configuration & Styling
st.set_page_config(
    page_title="Local RAG AI Assistant V2.1",
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
    
    .source-header {
        color: #3B82F6;
        font-weight: 600;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        margin-bottom: 0.3rem;
    }
    
    /* Session List Item */
    .session-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.4rem;
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .session-item-active {
        background: rgba(59, 130, 246, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    .tag-badge {
        display: inline-block;
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        padding: 2px 8px;
        border-radius: 20px;
        font-size: 0.75rem;
        margin-right: 4px;
        border: 1px solid rgba(59, 130, 246, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Helper function to run async RAG pipeline
async def run_local_rag(query: str, top_k: int = 2, file_type_filter: str = None):
    filter_val = None if file_type_filter == "Hepsi" else file_type_filter
    chunks = retrieve_context(query, top_k=top_k, file_type_filter=filter_val)
    
    if not chunks:
        return "I cannot find any relevant documents matching the query/filter in the database.", []
        
    context_text = ""
    for idx, chunk in enumerate(chunks, 1):
        context_text += f"\n[Doküman {idx}] (Kaynak: {chunk['title']})\n{chunk['content']}\n"
        
    # Local LLM (phi-4-mini) başlatma ve çıkarım
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
    
    model.unload()
    manager.stop_web_service()
    
    return answer, chunks

# Streamlit Session State Initialization
if "selected_session_id" not in st.session_state:
    # Varsayılan ilk oturumu oluştur
    sessions = get_sessions()
    if sessions:
        st.session_state.selected_session_id = sessions[0]["id"]
    else:
        initial_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        create_session(initial_id, f"Yeni Sohbet - {datetime.now().strftime('%H:%M')}")
        st.session_state.selected_session_id = initial_id

# Aktif oturum mesajlarını yükle
st.session_state.messages = get_session_messages(st.session_state.selected_session_id)

# Main Layout Headers
st.markdown("<h1 class='main-title'>🤖 Local RAG AI Assistant V2.1</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>SQLite Sohbet Geçmişi ve Yapay Zeka Özetleme Destekli Çevrimdışı Bilgi Sistemi</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<div class='sidebar-logo'>⚡ FOUNDRY LOCAL V2.1</div>", unsafe_allow_html=True)
    
    # Yeni Sohbet ve Oturum Yönetimi
    st.markdown("### 💬 Sohbetler")
    if st.button("➕ Yeni Sohbet Başlat", use_container_width=True):
        new_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        create_session(new_id, f"Sohbet - {datetime.now().strftime('%H:%M:%S')}")
        st.session_state.selected_session_id = new_id
        st.session_state.messages = []
        st.rerun()
        
    # Sohbet Listesi
    st.markdown("<div style='max-height: 200px; overflow-y: auto;'>", unsafe_allow_html=True)
    for session in get_sessions():
        is_active = session["id"] == st.session_state.selected_session_id
        active_class = "session-item-active" if is_active else ""
        
        # Flex layout ile silme butonunu hizalama
        col_title, col_del = st.columns([0.8, 0.2])
        with col_title:
            if st.button(session["title"], key=f"select_{session['id']}", use_container_width=True, type="secondary" if not is_active else "primary"):
                st.session_state.selected_session_id = session["id"]
                st.session_state.messages = get_session_messages(session["id"])
                st.rerun()
        with col_del:
            if st.button("🗑️", key=f"del_{session['id']}", help="Sohbeti sil", use_container_width=True):
                delete_session(session["id"])
                # Eğer silinen aktif sohbet ise başkasına geç
                if is_active:
                    remaining = get_sessions()
                    if remaining:
                        st.session_state.selected_session_id = remaining[0]["id"]
                    else:
                        st.session_state.pop("selected_session_id", None)
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Ayarlar & Dosyalar")
    
    # Arama Filtrelemesi (Metadata Filtering)
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1rem;font-weight:700;'>🔍 Filtre (Metaveri)</span>", unsafe_allow_html=True)
    filter_options = ["Hepsi", "text", "pdf", "docx", "python", "javascript", "markdown"]
    selected_filter = st.selectbox(
        "Tür:",
        options=filter_options,
        index=0,
        label_visibility="collapsed"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Dosya Yükleyici (Ingestion)
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1rem;font-weight:700;'>📁 Yeni Doküman / Kod Yükle</span>", unsafe_allow_html=True)
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
                
            with st.spinner("İçerik akıllıca parçalanıp özetleniyor (Çevrimdışı)..."):
                run_ingestion()
            st.success(f"'{uploaded_file.name}' indekslendi!")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# Main Application Tabs
tab_chat, tab_analytics = st.tabs(["💬 Sohbet Asistanı", "📊 Veri Analitiği (Dashboard)"])

# TAB 1: Chat Assistant
with tab_chat:
    # Karşılama Kartı
    if not st.session_state.messages:
        st.markdown("""
        <div class='welcome-card'>
            <div class='welcome-header'>👋 Çevrimdışı Bilgi Sistemine Hoş Geldiniz (V2.1)!</div>
            <div class='welcome-text'>
                Bu sürümde, sohbetleriniz <b>SQLite veritabanında saklanır</b> ve sol menüden geçmiş sohbetlerinize erişebilirsiniz.
                Arama sonuçlarındaki kod parçaları artık <b>akıllı kod renklendirici (Syntax Highlighting)</b> ile gösterilmektedir.
                <br><br>
                <b>Başlamak için bir soru yazabilir veya yeni sohbet açabilirsiniz.</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Sohbet Geçmişini Görüntüle
    for msg in st.session_state.messages:
        avatar = "👤" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Kullanıcı Girişi
    if user_query := st.chat_input("Yerel belgeleriniz hakkında soru sorun..."):
        # Kullanıcı mesajını ekle ve veritabanına kaydet
        st.chat_message("user", avatar="👤").markdown(user_query)
        save_message(st.session_state.selected_session_id, "user", user_query)
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Yerel model arama yapıyor ve cevap hazırlıyor (Çevrimdışı)..."):
                answer, chunks = asyncio.run(run_local_rag(user_query, file_type_filter=selected_filter))
                
                st.markdown(answer)
                
                # Akıllı Kod Renklendirmeli Kaynak Gösterimi
                if chunks:
                    with st.expander("🔍 Alakalı Kaynak Parçalarını Göster"):
                        for chunk in chunks:
                            st.markdown(f"<div class='source-header'>Kaynak: {chunk['title']} (RRF Skoru: {chunk['score']:.4f}, Tür: {chunk.get('file_type', 'txt').upper()})</div>", unsafe_allow_html=True)
                            
                            # Kod ise renklendirerek göster, metin ise info kartı olarak
                            ftype = chunk.get("file_type", "txt")
                            if ftype in ["python", "javascript", "markdown"]:
                                lang = "python" if ftype == "python" else ("javascript" if ftype == "javascript" else "markdown")
                                st.code(chunk["content"], language=lang)
                            else:
                                st.info(chunk["content"])
                                
                # Yanıtı veritabanına kaydet
                save_message(st.session_state.selected_session_id, "assistant", answer)
                
                # UI'ı tazelemek için tetikle
                st.rerun()

# TAB 2: Analytics Dashboard (Yapay Zeka Özetleri ile)
with tab_analytics:
    st.markdown("### 📊 Veritabanı ve Kütüphane İstatistikleri")
    
    db_docs = get_all_documents()
    
    if db_docs:
        df = pd.DataFrame(db_docs)
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
        chart_data = df["file_type"].value_counts()
        st.bar_chart(chart_data)
        
        # Doküman Listesi Tablosu (AI Özetleri ve Etiketleri ile)
        st.markdown("#### 📂 İndekslenen Dosyalar ve Yapay Zeka Analizi")
        
        # Dosya bazında grupla ve özet/etiket bilgilerini çek
        grouped = df.groupby("source_file").first().reset_index()
        
        for idx, row in grouped.iterrows():
            with st.container():
                st.markdown(f"##### 📄 {row['source_file']} (`{row['file_type'].upper()}`)")
                
                # Etiketleri rozet (badge) haline getir
                tags_html = ""
                if row['tags']:
                    for tag in row['tags'].split(","):
                        tags_html += f"<span class='tag-badge'>{tag.strip()}</span>"
                
                st.markdown(f"**Yapay Zeka Özeti:** {row['summary']}")
                st.markdown(f"**Etiketler:** {tags_html}", unsafe_allow_html=True)
                st.markdown("---")
        
    else:
        st.warning("Veritabanı boş. Lütfen sol menüden dosya yükleyin veya ingestion.py çalıştırın.")
