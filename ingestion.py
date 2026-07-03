import os
import sqlite3
import openai
from urllib.parse import urlparse
from database import init_db, insert_document, DB_FILE
from vector_search import VectorSearchEngine
from foundry_local_sdk import Configuration, FoundryLocalManager

DOCS_DIR = "documents"

def clear_database():
    """Veritabanındaki tüm kayıtları temizler."""
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

def extract_csv_text(file_path: str) -> str:
    import pandas as pd
    try:
        df = pd.read_csv(file_path)
        rows_text = []
        for idx, row in df.iterrows():
            row_str = ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
            rows_text.append(f"Row {idx+1}: {row_str}")
        return "\n\n".join(rows_text)
    except Exception as e:
        print(f"[HATA] CSV okunamadı: {e}")
        return ""

def extract_xlsx_text(file_path: str) -> str:
    import pandas as pd
    try:
        xls = pd.ExcelFile(file_path)
        sheets_text = []
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet_name)
            rows_text = [f"Sheet: {sheet_name}"]
            for idx, row in df.iterrows():
                row_str = ", ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                rows_text.append(f"Row {idx+1}: {row_str}")
            sheets_text.append("\n".join(rows_text))
        return "\n\n".join(sheets_text)
    except Exception as e:
        print(f"[HATA] Excel okunamadı: {e}")
        return ""


# --- Web Kazıyıcı (Scraper) Modülü ---

def scrape_and_save_url(url: str) -> str:
    """Belirtilen URL'den temiz metin çeker ve documents klasörüne kaydeder."""
    import requests
    from bs4 import BeautifulSoup
    
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        print(f"[SCRAPER] Adres kazınıyor: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Gereksiz script, style ve navigasyon öğelerini sil
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
            
        title = soup.title.string.strip() if soup.title else "Scraped Content"
        paragraphs = [p.get_text().strip() for p in soup.find_all("p") if p.get_text().strip()]
        
        full_text = f"Title: {title}\nURL: {url}\n\n" + "\n\n".join(paragraphs)
        
        # Dosya adı üret (örn: web_scraper_google_com.txt)
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.replace(".", "_")
        file_name = f"web_scraper_{domain}.txt"
        
        os.makedirs(DOCS_DIR, exist_ok=True)
        file_path = os.path.join(DOCS_DIR, file_name)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_text)
            
        print(f"[SCRAPER] Başarıyla kaydedildi: {file_path}")
        return file_name
    except Exception as e:
        print(f"[HATA] Web kazıma sırasında hata oluştu: {e}")
        return None

# --- Parçalama (Chunking) Yöntemleri ---

def split_into_child_chunks(text: str, size: int = 250) -> list[str]:
    """Ebeveyn metni yaklaşık *size* karakterlik küçük alt parçalara böler."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        if current_length >= size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return [c for c in chunks if c.strip()]

def get_parent_chunks_python(code: str) -> list[str]:
    """Python kodundan büyük sınıf ve fonksiyon ebeveyn parçaları çıkarır."""
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

def get_parent_chunks_javascript(code: str) -> list[str]:
    """JavaScript kodundan büyük fonksiyon ve sınıf ebeveyn parçaları çıkarır."""
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

def get_parent_chunks_markdown(md: str) -> list[str]:
    """Markdown metninden büyük başlık ebeveyn parçaları çıkarır."""
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

def get_parent_chunks_text(text: str) -> list[str]:
    """Düz metinleri büyük paragraflara göre böler (Ebeveyn parçalar)."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]

# --- AI ile Otomatik Özetleme ve Etiketleme ---

def generate_summary_and_tags(text: str, filename: str, endpoint: str, model_id: str) -> tuple[str, str]:
    client = openai.OpenAI(base_url=f"{endpoint}/v1", api_key="not-needed")
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
        print(f"   [HATA] Özet üretilemedi: {e}")
        return "Summary generation failed.", "failed"

# --- Ana İndeksleme Fonksiyonu ---

def run_ingestion():
    print("=== V3.0 DOKÜMAN VE KOD İNDEKSLEME BAŞLATILDI ===")
    init_db()
    clear_database()
    
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR, exist_ok=True)
        
    all_files = [f for f in os.listdir(DOCS_DIR) if os.path.splitext(f)[1].lower() in [".txt", ".pdf", ".docx", ".py", ".js", ".md", ".csv", ".xlsx"]]
    print(f"[SİSTEM] '{DOCS_DIR}' klasöründe {len(all_files)} dosya bulundu.")
    
    if not all_files:
        return
        
    # --- ADIM 1: phi-4-mini İLE ÖZETLERİ ÜRET ---
    print("\n[ADIM 1] Dosya özetleri ve etiketleri phi-4-mini ile üretiliyor...")
    web_config = Configuration.WebService(urls="http://127.0.0.1:0")
    config = Configuration(app_name="foundry-local-test", web=web_config)
    try:
        FoundryLocalManager.initialize(config)
    except Exception:
        pass
    manager = FoundryLocalManager.instance
    manager.start_web_service()
    endpoint = manager.urls[0]
    
    llm = manager.catalog.get_model("phi-4-mini")
    llm.load()
    
    file_summaries = {}
    file_contents = {}
    
    for file_name in all_files:
        file_path = os.path.join(DOCS_DIR, file_name)
        ext = os.path.splitext(file_name)[1].lower()
        try:
            if ext == ".pdf":
                text = extract_pdf_text(file_path)
            elif ext == ".docx":
                text = extract_docx_text(file_path)
            elif ext == ".csv":
                text = extract_csv_text(file_path)
            elif ext == ".xlsx":
                text = extract_xlsx_text(file_path)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            file_contents[file_name] = text
        except Exception as e:
            print(f" [HATA] {file_name} okunamadı: {e}")
            continue
            
        summary, tags = generate_summary_and_tags(text, file_name, endpoint, llm.id)
        file_summaries[file_name] = {"summary": summary, "tags": tags}
        
    llm.unload()
    manager.stop_web_service()
    
    # --- ADIM 2: EBEVEYN-ÇOCUK EMBEDDINGS ÜRET VE KAYDET ---
    print("\n[ADIM 2] Ebeveyn-Çocuk ilişkisiyle veri tabanı indeksleniyor...")
    search_engine = VectorSearchEngine()
    
    supported_extensions = {
        ".txt": "text",
        ".pdf": "pdf",
        ".docx": "docx",
        ".py": "python",
        ".js": "javascript",
        ".md": "markdown",
        ".csv": "csv",
        ".xlsx": "excel"
    }
    
    total_parents = 0
    total_children = 0
    
    for file_name, text in file_contents.items():
        ext = os.path.splitext(file_name)[1].lower()
        file_type = supported_extensions[ext]
        summary = file_summaries[file_name]["summary"]
        tags = file_summaries[file_name]["tags"]
        
        # 1. Ebeveyn parçaları çıkar (Parent Chunks)
        if file_type == "python":
            parent_chunks = get_parent_chunks_python(text)
        elif file_type == "javascript":
            parent_chunks = get_parent_chunks_javascript(text)
        elif file_type == "markdown":
            parent_chunks = get_parent_chunks_markdown(text)
        else:
            parent_chunks = get_parent_chunks_text(text)
            
        print(f"\nİndeksleniyor: {file_name} ({len(parent_chunks)} ana paragraf/ebeveyn)")
        
        for p_idx, p_chunk in enumerate(parent_chunks, 1):
            # Ebeveyni veritabanına kaydet (Vektörü NULL, is_parent=1)
            parent_title = f"{file_name} - Parent {p_idx}"
            parent_db_id = insert_document(
                title=parent_title,
                content=p_chunk,
                embedding=None,
                file_type=file_type,
                source_file=file_name,
                summary=summary,
                tags=tags,
                parent_id=None,
                is_parent=1
            )
            total_parents += 1
            
            # 2. Ebeveyn parçayı çocuk parçalara böl (Child Chunks)
            child_chunks = split_into_child_chunks(p_chunk, size=250)
            
            for c_idx, c_chunk in enumerate(child_chunks, 1):
                # Çocuk parçasını kaydet (Embedding'i var, is_parent=0, parent_id = parent_db_id)
                child_title = f"{file_name} - Child {p_idx}_{c_idx}"
                embedding = search_engine.generate_embedding(c_chunk)
                insert_document(
                    title=child_title,
                    content=c_chunk,
                    embedding=embedding,
                    file_type=file_type,
                    source_file=file_name,
                    summary=summary,
                    tags=tags,
                    parent_id=parent_db_id,
                    is_parent=0
                )
                total_children += 1
                
    search_engine.close()
    print(f"\n[BAŞARI] İndeksleme bitti. Ebeveyn: {total_parents}, Çocuk (Vektör): {total_children}")

if __name__ == "__main__":
    run_ingestion()
