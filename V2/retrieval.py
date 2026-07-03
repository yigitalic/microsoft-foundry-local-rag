import math
import re
from database import get_all_documents
from vector_search import VectorSearchEngine, cosine_similarity

# --- Basit ve Hızlı TF-IDF Kelime Arama Motoru ---

def tokenize(text: str) -> list[str]:
    """Metni kelimelere böler, küçük harfe çevirir ve temizler."""
    return re.findall(r'\b\w+\b', text.lower())

def compute_tfidf_scores(query: str, docs: list[dict]) -> dict:
    """Metinler arasında sorgu için TF-IDF kelime eşleşme skorlarını hesaplar."""
    query_words = tokenize(query)
    if not query_words:
        return {doc["id"]: 0.0 for doc in docs}
        
    total_docs = len(docs)
    
    # 1. Kelimelerin hangi dökümanlarda geçtiğini hesapla (IDF için)
    doc_contains = {}
    for word in query_words:
        doc_contains[word] = sum(1 for doc in docs if word in tokenize(doc["content"]))
        
    # 2. IDF değerlerini hesapla
    idf = {}
    for word in query_words:
        contains = doc_contains[word]
        # Sıfıra bölme hatasını önlemek için smoothing
        idf[word] = math.log(1 + (total_docs / (1 + contains)))
        
    scores = {}
    # 3. Her döküman için TF-IDF skorunu topla
    for doc in docs:
        doc_tokens = tokenize(doc["content"])
        doc_len = len(doc_tokens)
        
        doc_score = 0.0
        if doc_len > 0:
            for word in query_words:
                tf = doc_tokens.count(word) / doc_len
                doc_score += tf * idf[word]
                
        scores[doc["id"]] = doc_score
        
    return scores

# --- Reciprocal Rank Fusion (RRF) Birleştirme Algoritması ---

def reciprocal_rank_fusion(semantic_results: list[dict], lexical_results: list[dict], k: int = 60) -> list[dict]:
    """İki farklı sıralı arama sonucunu RRF skoru ile harmanlar."""
    # Her döküman kimliği (id) için RRF skor tablosu
    rrf_scores = {}
    
    # Yardımcı döküman haritası
    doc_map = {}
    
    # Anlamsal (Vektör) arama sıralaması
    for rank, doc in enumerate(semantic_results, 1):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        
    # Kelime (Lexical) arama sıralaması
    for rank, doc in enumerate(lexical_results, 1):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        
    # RRF skoruna göre sıralı döküman listesi oluştur
    merged_results = []
    for doc_id, rrf_score in rrf_scores.items():
        doc_info = doc_map[doc_id].copy()
        # Vektör bilgisini temizleyip sadece skoru ekliyoruz
        if "embedding" in doc_info:
            del doc_info["embedding"]
        doc_info["score"] = rrf_score
        merged_results.append(doc_info)
        
    # En yüksek RRF skorundan en düşüğe doğru sırala
    merged_results.sort(key=lambda x: x["score"], reverse=True)
    return merged_results

# --- Ana Hibrit Retrieval (Geri Getirme) Fonksiyonu ---

def retrieve_context(query: str, top_k: int = 2, file_type_filter: str = None) -> list[dict]:
    """Sorgu için Vektör ve Kelime aramayı (Hybrid) birleştirerek en alakalı doküman parçalarını getirir."""
    # 1. Veritabanındaki tüm dökümanları oku
    all_docs = get_all_documents()
    
    if not all_docs:
        return []
        
    # Metaveri Filtreleme (Metadata Filtering)
    if file_type_filter and file_type_filter.strip():
        all_docs = [doc for doc in all_docs if doc["file_type"] == file_type_filter]
        if not all_docs:
            return []
            
    # --- A. ANLAMSAL (VEKTÖR) ARAMA ---
    # Arama motorunu başlat (modeli yükler)
    search_engine = VectorSearchEngine()
    query_vector = search_engine.generate_embedding(query)
    search_engine.close() # Bellek koruması için hemen kapatıyoruz
    
    semantic_scored = []
    for doc in all_docs:
        score = cosine_similarity(query_vector, doc["embedding"])
        doc_copy = doc.copy()
        doc_copy["score"] = score
        semantic_scored.append(doc_copy)
        
    # Vektör skoruna göre sırala
    semantic_scored.sort(key=lambda x: x["score"], reverse=True)
    
    # --- B. KELİME (LEXICAL) ARAMA ---
    lexical_scores = compute_tfidf_scores(query, all_docs)
    lexical_scored = []
    for doc in all_docs:
        doc_copy = doc.copy()
        doc_copy["score"] = lexical_scores[doc["id"]]
        lexical_scored.append(doc_copy)
        
    # Kelime skoruna göre sırala
    lexical_scored.sort(key=lambda x: x["score"], reverse=True)
    
    # --- C. HİBRİT BİRLEŞTİRME (RRF) ---
    hybrid_results = reciprocal_rank_fusion(semantic_scored, lexical_scored)
    
    return hybrid_results[:top_k]

if __name__ == "__main__":
    # Test araması
    query = "Foundry Local'in desteklediği hızlandırıcılar"
    print(f"Test Hibrit Sorgu: '{query}'")
    context_chunks = retrieve_context(query, top_k=2)
    
    print("\nHibrit Arama Sonuçları (RRF Skoru):")
    for i, chunk in enumerate(context_chunks, 1):
        print(f"\n{i}. [{chunk['title']}] (RRF Skoru: {chunk['score']:.4f}, Tür: {chunk['file_type'].upper()})")
        print(f"   Metin: {chunk['content']}")
