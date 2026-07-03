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
from retrieval import retrieve_context, generate_expanded_queries
from ingestion import run_ingestion, scrape_and_save_url
from evaluator import run_evaluation_for_ui

# Veritabanını ilklendir
init_db()

# Page Configuration & Styling
st.set_page_config(
    page_title="Local RAG AI Assistant V3.0",
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
async def run_local_rag(query: str, top_k: int = 2, file_type_filter: str = None, use_reranker: bool = False):
    from foundry_local_sdk import Configuration, FoundryLocalManager
    
    # 1. Local LLM (phi-4-mini) servisini başlat (Sorgu genişletme ve arama için)
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
    
    # --- ADIM A: ÇOKLU SORGU GENİŞLETME (Multi-Query Expansion) ---
    st.toast("🔍 Sorgu genişletiliyor...")
    expanded_queries = generate_expanded_queries(query, client, model.id)
    # Genişletilen sorguları arayüzde bilgilendirme olarak göster
    if len(expanded_queries) > 1:
        st.toast(f"Genişletilen Sorgular: {', '.join(expanded_queries[1:])}")
        
    # --- ADIM B: HİBRİT EBEVEYN DÖKÜMAN ARAMASI (Parent-Document Retrieval) ---
    filter_val = None if file_type_filter == "Hepsi" else file_type_filter
    chunks = retrieve_context(query, top_k=top_k, file_type_filter=filter_val, expanded_queries=expanded_queries, client=client, model_id=model.id, use_reranker=use_reranker)
    
    if not chunks:
        model.unload()
        manager.stop_web_service()
        return "I cannot find any relevant documents matching the query/filter in the database.", []
        
    # Ebeveyn metinlerini birleştir
    context_text = ""
    for idx, chunk in enumerate(chunks, 1):
        context_text += f"\n[Doküman {idx}] (Kaynak: {chunk['title']})\n{chunk['content']}\n"
        
    # --- ADIM C: CEVAP ÜRETİMİ ---
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
if "selected_session_id" not in st.session_state:
    sessions = get_sessions()
    if sessions:
        st.session_state.selected_session_id = sessions[0]["id"]
    else:
        initial_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        create_session(initial_id, f"Sohbet - {datetime.now().strftime('%H:%M')}")
        st.session_state.selected_session_id = initial_id

st.session_state.messages = get_session_messages(st.session_state.selected_session_id)

# Main Layout Headers
st.markdown("<h1 class='main-title'>🤖 Local RAG AI Assistant V3.0</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Süper Gelişmiş Ebeveyn-Çocuk Retrieval, Çoklu Sorgu ve Web Kazıyıcılı RAG Sistemi</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("<div class='sidebar-logo'>⚡ FOUNDRY LOCAL V3.0</div>", unsafe_allow_html=True)
    
    # Yeni Sohbet ve Oturum Yönetimi
    st.markdown("### 💬 Sohbet Oturumları")
    if st.button("➕ Yeni Sohbet Başlat", use_container_width=True):
        new_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        create_session(new_id, f"Sohbet - {datetime.now().strftime('%H:%M')}")
        st.session_state.selected_session_id = new_id
        st.session_state.messages = []
        st.rerun()
        
    # Sohbet Listesi
    st.markdown("<div style='max-height: 180px; overflow-y: auto; margin-bottom: 10px;'>", unsafe_allow_html=True)
    for session in get_sessions():
        is_active = session["id"] == st.session_state.selected_session_id
        active_class = "session-item-active" if is_active else ""
        col_title, col_del = st.columns([0.8, 0.2])
        with col_title:
            if st.button(session["title"], key=f"select_{session['id']}", use_container_width=True, type="secondary" if not is_active else "primary"):
                st.session_state.selected_session_id = session["id"]
                st.session_state.messages = get_session_messages(session["id"])
                st.rerun()
        with col_del:
            if st.button("🗑️", key=f"del_{session['id']}", use_container_width=True):
                delete_session(session["id"])
                if is_active:
                    remaining = get_sessions()
                    if remaining:
                        st.session_state.selected_session_id = remaining[0]["id"]
                    else:
                        st.session_state.pop("selected_session_id", None)
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("### ⚙️ Ayarlar & Dosya Yükleme")
    
    # Arama Filtrelemesi
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1rem;font-weight:700;'>🔍 Filtre (Metaveri)</span>", unsafe_allow_html=True)
    filter_options = ["Hepsi", "text", "pdf", "docx", "python", "javascript", "markdown", "csv", "excel"]
    selected_filter = st.selectbox(
        "Tür:", options=filter_options, index=0, label_visibility="collapsed"
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Reranker Ayarları
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1rem;font-weight:700;'>⚙️ Arama Parametreleri</span>", unsafe_allow_html=True)
    use_reranker = st.checkbox("Yerel Re-ranking (LLM) Aktif", value=False, help="Aday dokümanları yerel LLM ile yeniden puanlar ve sıralar.")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Dosya Yükleyici
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1rem;font-weight:700;'>📁 Belge / Kod Yükle</span>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Metin, PDF, Excel, CSV, Kod...",
        type=["txt", "pdf", "docx", "py", "js", "md", "csv", "xlsx"],
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        if st.button("İndeksle", key="upload_btn", use_container_width=True):
            os.makedirs("documents", exist_ok=True)
            file_path = os.path.join("documents", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner("AI Özet çıkarılıyor ve parçalanıyor..."):
                run_ingestion()
            st.success(f"'{uploaded_file.name}' indekslendi!")
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Web Scraper (URL Kazıma)
    st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
    st.markdown("<span style='font-size:1rem;font-weight:700;'>🌐 Web Sayfası Kazı & İndeksle</span>", unsafe_allow_html=True)
    url_input = st.text_input("URL girin (http...):", label_visibility="collapsed")
    if st.button("Web Sayfasını İndeksle", use_container_width=True):
        if url_input.strip().startswith("http"):
            with st.spinner("Web sayfası indiriliyor ve temizleniyor..."):
                file_name = scrape_and_save_url(url_input.strip())
                if file_name:
                    run_ingestion()
                    st.success(f"Web sayfası '{file_name}' olarak indekslendi!")
                    st.rerun()
                else:
                    st.error("Web sayfası kazınamadı.")
        else:
            st.error("Lütfen geçerli bir URL girin.")
    st.markdown("</div>", unsafe_allow_html=True)

# Main Application Tabs
tab_chat, tab_analytics, tab_eval = st.tabs(["💬 Sohbet Asistanı", "📊 Veri Analitiği (Dashboard)", "🧪 Sistem Değerlendirme (Evaluator)"])

# TAB 1: Chat Assistant
with tab_chat:
    if not st.session_state.messages:
        st.markdown("""
        <div class='welcome-card'>
            <div class='welcome-header'>👋 Süper Gelişmiş Yerel RAG Sistemine Hoş Geldiniz (V3.0)!</div>
            <div class='welcome-text'>
                Bu sürümde, arama kalitesini artırmak için şu teknolojiler bir arada çalışır:
                <ul>
                    <li><b>Multi-Query Expansion:</b> Sorduğunuz soru arka planda yerel modelle çoğaltılarak aranır.</li>
                    <li><b>Parent-Document Retrieval:</b> Arama küçük parçalar üzerinde yapılır, ancak yapay zekaya dökümanın ebeveyn paragrafı beslenir.</li>
                    <li><b>Web Scraper:</b> Sol menüden bir URL girerek web sayfalarını anında veritabanınıza katabilirsiniz.</li>
                </ul>
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
        st.chat_message("user", avatar="👤").markdown(user_query)
        save_message(st.session_state.selected_session_id, "user", user_query)
        
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Sorgu genişletiliyor, aranıyor ve cevaplanıyor (Çevrimdışı)..."):
                answer, chunks = asyncio.run(run_local_rag(user_query, file_type_filter=selected_filter, use_reranker=use_reranker))
                
                st.markdown(answer)
                
                # Akıllı Kod Renklendirmeli Kaynak Gösterimi
                if chunks:
                    with st.expander("🔍 Alakalı Ebeveyn Kaynakları Göster"):
                        for chunk in chunks:
                            st.markdown(f"<div class='source-header'>Kaynak: {chunk['title']} (RRF Skoru: {chunk['score']:.4f})</div>", unsafe_allow_html=True)
                            
                            ftype = chunk.get("file_type", "text")
                            if ftype in ["python", "javascript", "markdown"]:
                                lang = "python" if ftype == "python" else ("javascript" if ftype == "javascript" else "markdown")
                                st.code(chunk["content"], language=lang)
                            else:
                                st.info(chunk["content"])
                                
                save_message(st.session_state.selected_session_id, "assistant", answer)
                st.rerun()

# TAB 2: Analytics Dashboard
with tab_analytics:
    st.markdown("### 📊 Veritabanı ve Kütüphane İstatistikleri")
    
    db_docs = get_all_documents()
    
    if db_docs:
        df = pd.DataFrame(db_docs)
        # Sadece ebeveyn dökümanlar üzerinden dosya sayısı hesapla
        unique_files = df["source_file"].nunique()
        total_chunks = len(df)
        child_chunks = len(df[df["is_parent"] == 0])
        parent_chunks = len(df[df["is_parent"] == 1])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Toplam İndekslenen Belge", unique_files)
        with col2:
            st.metric("Ebeveyn Paragraf Sayısı", parent_chunks)
        with col3:
            st.metric("Alt Vektör Parçacıkları (Child)", child_chunks)
            
        # Dosya Türlerine Göre Dağılım Grafiği (Sadece çocuk parçalar üzerinden)
        st.markdown("#### 📈 Dosya Formatı Dağılımı")
        chart_data = df[df["is_parent"] == 0]["file_type"].value_counts()
        st.bar_chart(chart_data)
        
        # Yapay Zeka Özetleri
        st.markdown("#### 📂 İndekslenen Dosyalar ve Yapay Zeka Analizi")
        grouped = df.groupby("source_file").first().reset_index()
        
        for idx, row in grouped.iterrows():
            with st.container():
                st.markdown(f"##### 📄 {row['source_file']} (`{row['file_type'].upper()}`)")
                
                tags_html = ""
                if row['tags']:
                    for tag in row['tags'].split(","):
                        tags_html += f"<span class='tag-badge'>{tag.strip()}</span>"
                
                st.markdown(f"**Yapay Zeka Özeti:** {row['summary']}")
                st.markdown(f"**Etiketler:** {tags_html}", unsafe_allow_html=True)
                st.markdown("---")
        
    else:
        st.warning("Veritabanı boş. Lütfen sol menüden dosya yükleyin veya ingestion.py çalıştırın.")

# TAB 3: System Evaluation (Evaluator UI)
with tab_eval:
    st.markdown("### 🧪 Yerel RAG Değerlendirme ve Kalite Raporu")
    st.markdown("Bu panelde, RAG sistemini 3 farklı kategoride (Bilgi Tabanı, Kod/Teknik Soru, Sınır Durum) test ederek sistemin **Doğruluk (Faithfulness)** ve **Alaka (Relevance)** düzeylerini yerel yapay zeka hakemliği (LLM-as-a-Judge) ile değerlendirebilirsiniz.")
    
    if st.button("🚀 Değerlendirmeyi Başlat", use_container_width=True):
        with st.spinner("Yerel model yükleniyor ve test senaryoları koşturuluyor..."):
            from foundry_local_sdk import Configuration, FoundryLocalManager
            from evaluator import run_evaluation_for_ui
            
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
            
            try:
                eval_results = run_evaluation_for_ui(client, model.id)
                
                # Metrik hesapla
                avg_faithfulness = sum(r["faithfulness"] for r in eval_results) / len(eval_results)
                avg_relevance = sum(r["relevance"] for r in eval_results) / len(eval_results)
                
                st.success("Değerlendirme başarıyla tamamlandı!")
                
                # Metrik kartları
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Ortalama Doğruluk (Faithfulness)", f"{avg_faithfulness:.1f} / 5.0")
                with col2:
                    st.metric("Ortalama Alaka (Relevance)", f"{avg_relevance:.1f} / 5.0")
                    
                # Ayrıntılı kartlar
                st.markdown("#### 📋 Senaryo Detayları")
                for idx, res in enumerate(eval_results, 1):
                    with st.container():
                        st.markdown(f"##### **Senaryo {idx}: {res['category']}**")
                        st.markdown(f"**Soru:** `{res['question']}`")
                        
                        # Kaynaklar
                        srcs_html = " ".join([f"<span class='tag-badge'>{s}</span>" for s in res['sources']])
                        st.markdown(f"**Bulunan Kaynaklar:** {srcs_html}", unsafe_allow_html=True)
                        
                        # Cevap
                        st.markdown(f"**Üretilen Cevap:**")
                        st.info(res['answer'])
                        
                        # LLM Judge Puanları
                        col_f, col_r = st.columns(2)
                        with col_f:
                            st.markdown(f"**Doğruluk Puanı:** `{res['faithfulness']}/5` 🎯")
                        with col_r:
                            st.markdown(f"**Alaka Puanı:** `{res['relevance']}/5` 👁️")
                            
                        st.markdown(f"**Hakem Açıklaması:** *{res['explanation']}*")
                        st.markdown("---")
                        
            except Exception as e:
                st.error(f"Değerlendirme sırasında bir hata oluştu: {e}")
            finally:
                model.unload()
                manager.stop_web_service()
