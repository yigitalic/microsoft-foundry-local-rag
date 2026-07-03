from database import get_all_documents
from vector_search import VectorSearchEngine, cosine_similarity

def retrieve_context(query: str, top_k: int = 2) -> list[dict]:
    """Sorgu için en alakalı doküman parçalarını (context) veritabanından getirir."""
    # Arama motorunu başlat (modeli yükler)
    search_engine = VectorSearchEngine()
    
    # 1. Sorgunun embedding vektörünü üret
    query_vector = search_engine.generate_embedding(query)
    
    # 2. Veritabanındaki tüm dökümanları oku
    db_docs = get_all_documents()
    
    results = []
    # 3. Her bir döküman için benzerlik hesapla
    for doc in db_docs:
        score = cosine_similarity(query_vector, doc["embedding"])
        results.append({
            "title": doc["title"],
            "content": doc["content"],
            "score": score
        })
        
    # 4. Skora göre azalan şekilde sırala ve en üstteki top_k adet kaydı seç
    results.sort(key=lambda x: x["score"], reverse=True)
    top_results = results[:top_k]
    
    # Arama motorunu kapat
    search_engine.close()
    
    return top_results

if __name__ == "__main__":
    # Basit bir test sorgusu
    query = "Staj sunumunun süresi kaç dakikadır?"
    print(f"Test Arama Sorgusu: '{query}'")
    context_chunks = retrieve_context(query, top_k=2)
    
    print("\nBulunan En Alakalı Parçalar:")
    for i, chunk in enumerate(context_chunks, 1):
        print(f"\n{i}. [{chunk['title']}] (Benzerlik Skoru: {chunk['score']:.4f})")
        print(f"   Metin: {chunk['content']}")
