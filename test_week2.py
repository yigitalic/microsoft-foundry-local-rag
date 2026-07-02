import json
import sqlite3
from database import init_db, insert_document, get_all_documents, DB_FILE
from vector_search import VectorSearchEngine, cosine_similarity

def clear_database():
    """Test için veritabanını temizler."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM documents")
    conn.commit()
    conn.close()
    print("[SİSTEM] Test öncesi veritabanı temizlendi.")

def main():
    print("=== 2. HAFTA ENTEGRASYON VE DOĞRULAMA TESTİ ===")
    
    # 1. Veritabanını başlat ve temizle
    init_db()
    clear_database()
    
    # 2. Vektör arama motorunu başlat (Embedding modelini yükler)
    search_engine = VectorSearchEngine()
    
    # Test dokümanları
    sample_docs = [
        {
            "title": "Microsoft Foundry Local",
            "content": "Microsoft Foundry Local allows you to download and run large language models entirely offline on your personal computer without needing cloud credits or internet."
        },
        {
            "title": "SQLite Database",
            "content": "SQLite is a serverless, self-contained SQL database engine that stores all tables and data in a single file on disk, making it ideal for local storage."
        },
        {
            "title": "Retrieval-Augmented Generation (RAG)",
            "content": "Retrieval-Augmented Generation (RAG) is an AI architecture that retrieves relevant documents from a database and uses them as context for LLM response generation."
        }
    ]
    
    # 3. Dokümanları embedding'leri ile veritabanına kaydet
    print("\n[ADIM 1] Dokümanlar yerel modelle embedding üretilerek SQLite veritabanına kaydediliyor...")
    for doc in sample_docs:
        print(f" - Vektör üretiliyor: '{doc['title']}'")
        emb = search_engine.generate_embedding(doc["content"])
        insert_document(doc["title"], doc["content"], emb)
    print("Kayıt işlemi tamamlandı.")
    
    # 4. Veritabanındaki tüm kayıtları oku
    db_docs = get_all_documents()
    print(f"\n[ADIM 2] Veritabanından toplam {len(db_docs)} kayıt başarıyla okundu.")
    
    # 5. Arama sorgusu sor ve benzerlik karşılaştırması yap
    query = "How can I run AI models offline on my computer?"
    print(f"\n[ADIM 3] Arama sorgusu: '{query}'")
    
    # Sorgunun embedding vektörünü üret
    query_vector = search_engine.generate_embedding(query)
    
    results = []
    # Veritabanındaki tüm dokümanlarla sorgu benzerliğini hesapla
    for doc in db_docs:
        score = cosine_similarity(query_vector, doc["embedding"])
        results.append({
            "title": doc["title"],
            "content": doc["content"],
            "score": score
        })
        
    # Sonuçları en benzerden (en yüksek skordan) en düşüğe göre sırala
    results.sort(key=lambda x: x["score"], reverse=True)
    
    print("\n=== ARAMA SONUÇLARI (En Alakalıya Göre Sıralı) ===")
    for i, res in enumerate(results, 1):
        print(f"{i}. [{res['title']}] (Benzerlik Skoru: {res['score']:.4f})")
        print(f"   Metin: {res['content']}\n")
        
    # 6. Kapatma temizliği
    search_engine.close()
    print("\nTest başarıyla sonlandırıldı.")

if __name__ == "__main__":
    main()
