import os
import sqlite3
from database import init_db, insert_document, DB_FILE
from vector_search import VectorSearchEngine

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
    """PDF dosyasından metin çıkarır."""
    import pypdf
    reader = pypdf.PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n\n"
    return text

def extract_docx_text(file_path: str) -> str:
    """Word (.docx) dosyasından metin çıkarır."""
    import docx
    doc = docx.Document(file_path)
    text = ""
    for p in doc.paragraphs:
        if p.text.strip():
            text += p.text + "\n\n"
    return text

# --- Dosya Tiplerine Göre Akıllı Parçalama (Chunking) Yöntemleri ---

def chunk_python_code(code: str) -> list[str]:
    """Python kodunu sınıf (class) ve fonksiyon (def) yapılarına göre böler."""
    lines = code.split("\n")
    chunks = []
    current_chunk = []
    
    for line in lines:
        # Sınıf veya fonksiyon tanımı gördüğümüzde yeni parçaya geç
        if (line.startswith("class ") or line.startswith("def ")) and current_chunk:
            chunks.append("\n".join(current_chunk).strip())
            current_chunk = []
        current_chunk.append(line)
        
    if current_chunk:
        chunks.append("\n".join(current_chunk).strip())
    return [c for c in chunks if c]

def chunk_javascript_code(code: str) -> list[str]:
    """JavaScript kodunu fonksiyon ve sınıf yapılarına göre böler."""
    lines = code.split("\n")
    chunks = []
    current_chunk = []
    
    for line in lines:
        # function, class veya arrow function tanım satırlarında böl
        if (line.startswith("function ") or line.startswith("class ") or ("const " in line and "=>" in line)) and current_chunk:
            chunks.append("\n".join(current_chunk).strip())
            current_chunk = []
        current_chunk.append(line)
        
    if current_chunk:
        chunks.append("\n".join(current_chunk).strip())
    return [c for c in chunks if c]

def chunk_markdown(md: str) -> list[str]:
    """Markdown metnini başlıklara (#, ##, ###) göre böler."""
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
    """Düz metinleri paragraflara göre böler."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]

# --- Ana İndeksleme Fonksiyonu ---

def run_ingestion():
    print("=== V2 DOKÜMAN VE KOD İNDEKSLEME BAŞLATILDI ===")
    
    # 1. Veritabanını V2 şemasıyla ilklendir ve temizle
    init_db()
    clear_database()
    
    # 2. Vektör arama motorunu yükle
    search_engine = VectorSearchEngine()
    
    if not os.path.exists(DOCS_DIR):
        print(f"[HATA] '{DOCS_DIR}' klasörü bulunamadı. Oluşturuluyor...")
        os.makedirs(DOCS_DIR, exist_ok=True)
        
    all_files = os.listdir(DOCS_DIR)
    print(f"[SİSTEM] '{DOCS_DIR}' klasöründe {len(all_files)} adet dosya tespit edildi.")
    
    total_chunks = 0
    
    # Supported formats mapping
    supported_extensions = {
        ".txt": "text",
        ".pdf": "pdf",
        ".docx": "docx",
        ".py": "python",
        ".js": "javascript",
        ".md": "markdown"
    }
    
    # 3. Her dosyayı oku, türüne göre parçala ve kaydet
    for file_name in all_files:
        ext = os.path.splitext(file_name)[1].lower()
        if ext not in supported_extensions:
            print(f" - [ATLANIYOR] Desteklenmeyen dosya türü: {file_name}")
            continue
            
        file_type = supported_extensions[ext]
        file_path = os.path.join(DOCS_DIR, file_name)
        print(f"\nİşleniyor: {file_name} (Tür: {file_type.upper()})")
        
        # Metin okuma
        try:
            if file_type == "pdf":
                text = extract_pdf_text(file_path)
            elif file_type == "docx":
                text = extract_docx_text(file_path)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
        except Exception as e:
            print(f" [HATA] Dosya okunurken hata oluştu {file_name}: {e}")
            continue
            
        # Parçalama (Chunking)
        if file_type == "python":
            chunks = chunk_python_code(text)
        elif file_type == "javascript":
            chunks = chunk_javascript_code(text)
        elif file_type == "markdown":
            chunks = chunk_markdown(text)
        else:
            chunks = chunk_plain_text(text)
            
        print(f"-> {len(chunks)} adet anlamsal parça üretildi.")
        
        # Embedding üretme ve kaydetme
        for idx, chunk in enumerate(chunks, 1):
            title = f"{file_name} - Parça {idx}"
            print(f"   Vektör üretiliyor & Kaydediliyor {idx}/{len(chunks)}...")
            embedding = search_engine.generate_embedding(chunk)
            insert_document(
                title=title,
                content=chunk,
                embedding=embedding,
                file_type=file_type,
                source_file=file_name
            )
            total_chunks += 1
            
    print(f"\n[BAŞARI] Toplam {total_chunks} parça başarıyla V2 veritabanına indekslendi.")
    
    # Kapatma temizliği
    search_engine.close()

if __name__ == "__main__":
    run_ingestion()
