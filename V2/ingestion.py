import os
import sqlite3
import openai
from database import init_db, insert_document, DB_FILE
from vector_search import VectorSearchEngine
from foundry_local_sdk import Configuration, FoundryLocalManager

DOCS_DIR = "documents"

def clear_database():
    """Veritabanındaki tüm dökümanları temizler."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents")
    conn.commit()
    conn.close()
    print("[SİSTEM] Veritabanı temizlendi.")

# --- Dosya Tiplerine Göre Metin Çekme Fonksiyonları ---

def extract_pdf_text(file_path: str) -> str:
    import pypdf
    reader = pypdf.PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n\n"
    return text

def extract_docx_text(file_path: str) -> str:
    import docx
    doc = docx.Document(file_path)
    text = ""
    for p in doc.paragraphs:
        if p.text.strip():
            text += p.text + "\n\n"
    return text

# --- Dosya Tiplerine Göre Akıllı Parçalama (Chunking) Yöntemleri ---

def chunk_python_code(code: str) -> list[str]:
    lines = code.split("\n")
    chunks = []
    current_chunk = []
    
    for line in lines:
        if (line.startswith("class ") or line.startswith("def ")) and current_chunk:
            chunks.append("\n".join(current_chunk).strip())
            current_chunk = []
        current_chunk.append(line)
        
    if current_chunk:
        chunks.append("\n".join(current_chunk).strip())
    return [c for c in chunks if c]

def chunk_javascript_code(code: str) -> list[str]:
    lines = code.split("\n")
    chunks = []
    current_chunk = []
    
    for line in lines:
        if (line.startswith("function ") or line.startswith("class ") or ("const " in line and "=>" in line)) and current_chunk:
            chunks.append("\n".join(current_chunk).strip())
            current_chunk = []
        current_chunk.append(line)
        
    if current_chunk:
        chunks.append("\n".join(current_chunk).strip())
    return [c for c in chunks if c]

def chunk_markdown(md: str) -> list[str]:
    lines = md.split("\n")
    chunks = []
    current_chunk = []
    
    for line in lines:
        if (line.startswith("# ") or line.startswith("## ") or line.startswith("### ")) and current_chunk:
            chunks.append("\n".join(current_chunk).strip())
            current_chunk = []
        current_chunk.append(line)
        
    if current_chunk:
        chunks.append("\n".join(current_chunk).strip())
    return [c for c in chunks if c]

def chunk_plain_text(text: str) -> list[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]

# --- AI ile Otomatik Özetleme ve Etiketleme Yardımcısı ---

def generate_summary_and_tags(text: str, filename: str, endpoint: str, model_id: str) -> tuple[str, str]:
    """phi-4-mini kullanarak dökümana 1 cümlelik özet ve 5 adet etiket üretir."""
    client = openai.OpenAI(base_url=f"{endpoint}/v1", api_key="not-needed")
    
    # Bellek ve hız tasarrufu amacıyla ilk 2000 karakteri analiz ediyoruz
    text_sample = text[:2000]
    
    prompt = (
        f"Analyze the document '{filename}' and write a brief 1-sentence summary "
        "and exactly 5 comma-separated keywords/tags describing its topic.\n\n"
        "Document Content:\n"
        f"{text_sample}\n\n"
        "You MUST respond ONLY in the following format (no explanations, no other text):\n"
        "Summary: [your 1-sentence summary]\n"
        "Tags: [tag1, tag2, tag3, tag4, tag5]"
    )
    
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        output = response.choices[0].message.content.strip()
        
        summary = "No summary generated."
        tags = "general"
        
        for line in output.split("\n"):
            if line.lower().startswith("summary:"):
                summary = line.split(":", 1)[1].strip()
            elif line.lower().startswith("tags:"):
                tags = line.split(":", 1)[1].strip()
                
        return summary, tags
    except Exception as e:
        print(f"   [HATA] Özet üretilirken hata oluştu: {e}")
        return "Summary generation failed.", "failed"

# --- Ana İndeksleme Fonksiyonu ---

def run_ingestion():
    print("=== V2.1 DOKÜMAN VE KOD İNDEKSLEME BAŞLATILDI ===")
    
    # 1. Veritabanını ilklendir ve temizle
    init_db()
    clear_database()
    
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR, exist_ok=True)
        
    all_files = [f for f in os.listdir(DOCS_DIR) if os.path.splitext(f)[1].lower() in [".txt", ".pdf", ".docx", ".py", ".js", ".md"]]
    print(f"[SİSTEM] '{DOCS_DIR}' klasöründe {len(all_files)} adet indekslenecek dosya bulundu.")
    
    if not all_files:
        print("[SİSTEM] İndekslenecek dosya yok. İşlem tamamlandı.")
        return
        
    # --- ADIM A: phi-4-mini İLE ÖZETLERİ ÜRET (Çevrimdışı LLM) ---
    print("\n[ADIM 1] Dosya özetleri ve etiketleri phi-4-mini ile üretiliyor...")
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
    
    # Phi-4-mini yükle
    llm = catalog.get_model("phi-4-mini")
    llm.load()
    
    file_summaries = {}
    file_contents = {}
    
    for file_name in all_files:
        file_path = os.path.join(DOCS_DIR, file_name)
        ext = os.path.splitext(file_name)[1].lower()
        
        # Metni oku
        try:
            if ext == ".pdf":
                text = extract_pdf_text(file_path)
            elif ext == ".docx":
                text = extract_docx_text(file_path)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
        except Exception as e:
            print(f" [HATA] {file_name} okunamadı: {e}")
            continue
            
        file_contents[file_name] = text
        
        # Özet ve etiket üret
        print(f" - Özetleniyor: {file_name}...")
        summary, tags = generate_summary_and_tags(text, file_name, endpoint, llm.id)
        file_summaries[file_name] = {"summary": summary, "tags": tags}
        print(f"   -> Özet: {summary}")
        print(f"   -> Etiketler: {tags}")
        
    # LLM bellekten çıkarılır
    llm.unload()
    manager.stop_web_service()
    
    # --- ADIM B: qwen3-embedding İLE EMBEDDINGS ÜRET VE KAYDET ---
    print("\n[ADIM 2] Doküman parçaları embedding üretilerek SQLite veritabanına kaydediliyor...")
    search_engine = VectorSearchEngine()
    
    supported_extensions = {
        ".txt": "text",
        ".pdf": "pdf",
        ".docx": "docx",
        ".py": "python",
        ".js": "javascript",
        ".md": "markdown"
    }
    
    total_chunks = 0
    for file_name, text in file_contents.items():
        ext = os.path.splitext(file_name)[1].lower()
        file_type = supported_extensions[ext]
        
        # Parçalama (Chunking)
        if file_type == "python":
            chunks = chunk_python_code(text)
        elif file_type == "javascript":
            chunks = chunk_javascript_code(text)
        elif file_type == "markdown":
            chunks = chunk_markdown(text)
        else:
            chunks = chunk_plain_text(text)
            
        # Dosyaya ait özet ve etiketleri al
        summary = file_summaries[file_name]["summary"]
        tags = file_summaries[file_name]["tags"]
        
        print(f"\nKaydediliyor: {file_name} ({len(chunks)} parça)")
        for idx, chunk in enumerate(chunks, 1):
            title = f"{file_name} - Parça {idx}"
            embedding = search_engine.generate_embedding(chunk)
            insert_document(
                title=title,
                content=chunk,
                embedding=embedding,
                file_type=file_type,
                source_file=file_name,
                summary=summary,
                tags=tags
            )
            total_chunks += 1
            
    # Kapatma temizliği
    search_engine.close()
    print(f"\n[BAŞARI] Toplam {total_chunks} parça özet ve etiketleriyle birlikte SQLite veritabanına indekslendi.")

if __name__ == "__main__":
    run_ingestion()
