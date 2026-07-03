import os
from database import init_db, insert_document, get_all_documents, DB_FILE
import sqlite3
from vector_search import VectorSearchEngine

DOCS_DIR = "documents"

def clear_database():
    """Veritabanını temizler."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents")
    conn.commit()
    conn.close()
    print("[SİSTEM] Veritabanı temizlendi.")

def chunk_text(text: str) -> list[str]:
    """Metni çift satır aralıklarına (paragraflara) göre böler."""
    # Boş olmayan paragrafları ayıkla
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return paragraphs

def run_ingestion():
    print("=== DOKÜMAN VERİ GİRİŞİ (INGESTION) BAŞLATILDI ===")
    
    # 1. Veritabanını ilklendir ve temizle
    init_db()
    clear_database()
    
    # 2. Vektör arama motorunu başlat
    search_engine = VectorSearchEngine()
    
    if not os.path.exists(DOCS_DIR):
        print(f"[HATA] '{DOCS_DIR}' klasörü bulunamadı. Lütfen oluşturun.")
        search_engine.close()
        return
        
    files = [f for f in os.listdir(DOCS_DIR) if f.endswith(".txt")]
    print(f"[SİSTEM] '{DOCS_DIR}' klasöründe {len(files)} adet metin belgesi bulundu.")
    
    total_chunks = 0
    
    # 3. Her dosyayı oku ve parçala
    for file_name in files:
        file_path = os.path.join(DOCS_DIR, file_name)
        print(f"\nDosya okunuyor: {file_name}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        chunks = chunk_text(content)
        print(f"-> Doküman {len(chunks)} parçaya ayrıldı.")
        
        # 4. Her parçayı embedding'i ile kaydet
        for idx, chunk in enumerate(chunks, 1):
            title = f"{file_name} - Parça {idx}"
            print(f"   Vektör üretiliyor & Kaydediliyor {idx}/{len(chunks)}...")
            embedding = search_engine.generate_embedding(chunk)
            insert_document(title, chunk, embedding)
            total_chunks += 1
            
    print(f"\n[BAŞARI] Toplam {len(files)} dosyadan {total_chunks} parça başarıyla SQLite'a yüklendi.")
    
    # Kapatma temizliği
    search_engine.close()

if __name__ == "__main__":
    run_ingestion()
