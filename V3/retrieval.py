import math
import re
from database import get_all_documents, get_document_by_id
from vector_search import VectorSearchEngine, cosine_similarity

# --- TF-IDF Kelime Arama Motoru ---

def tokenize(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())

def compute_tfidf_scores(query: str, docs: list[dict]) -> dict:
    query_words = tokenize(query)
    if not query_words:
        return {doc["id"]: 0.0 for doc in docs}
        
    total_docs = len(docs)
    doc_contains = {}
    for word in query_words:
        doc_contains[word] = sum(1 for doc in docs if word in tokenize(doc["content"]))
        
    idf = {}
    for word in query_words:
        contains = doc_contains[word]
        idf[word] = math.log(1 + (total_docs / (1 + contains)))
        
    scores = {}
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

# --- Reciprocal Rank Fusion (RRF) ---

def reciprocal_rank_fusion(semantic_results: list[dict], lexical_results: list[dict], k: int = 60) -> list[dict]:
    rrf_scores = {}
    doc_map = {}
    
    for rank, doc in enumerate(semantic_results, 1):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        
    for rank, doc in enumerate(lexical_results, 1):
        doc_id = doc["id"]
        doc_map[doc_id] = doc
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
        
    merged_results = []
    for doc_id, rrf_score in rrf_scores.items():
        doc_info = doc_map[doc_id].copy()
        if "embedding" in doc_info:
            del doc_info["embedding"]
        doc_info["score"] = rrf_score
        merged_results.append(doc_info)
        
    merged_results.sort(key=lambda x: x["score"], reverse=True)
    return merged_results

# --- Multi-Query Genişletme Yardımcısı (LLM Kullanarak) ---

def generate_expanded_queries(query: str, client, model_id: str) -> list[str]:
    """phi-4-mini kullanarak sorguya benzer 2 alternatif arama sorgusu üretir."""
    prompt = (
        f"You are a search query optimizer. Generate exactly 2 alternative, semantically similar search queries "
        f"in Turkish (or the language of the query) based on this input: '{query}'.\n"
        "Your response must be ONLY the queries, one per line. No numbering, no introduction, no punctuation."
    )
    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        queries = [q.strip() for q in response.choices[0].message.content.strip().split("\n") if q.strip()]
        # Orijinal sorguyu da ekleyip listeyi döndür
        return [query] + queries[:2]
    except Exception as e:
        print(f"[HATA] Sorgu genişletilemedi: {e}")
        return [query]

# --- Ana Hibrit Parent-Document Retrieval Fonksiyonu ---

def retrieve_context(query: str, top_k: int = 2, file_type_filter: str = None, expanded_queries: list[str] = None) -> list[dict]:
    """
    Sorgu varyasyonlarını kullanarak alt (child) parçalar arasında arama yapar
    ve eşleşen en yüksek RRF skorlu parçaların ebeveyn (parent) dökümanlarını SQLite'tan çekip döndürür.
    """
    # 1. Veritabanındaki tüm dökümanları oku
    all_docs = get_all_documents()
    
    if not all_docs:
        return []
        
    # Metaveri Filtreleme
    if file_type_filter and file_type_filter.strip():
        all_docs = [doc for doc in all_docs if doc["file_type"] == file_type_filter]
        if not all_docs:
            return []
            
    # Arama sadece çocuk parçalar (is_parent=0) arasında yapılır
    child_docs = [doc for doc in all_docs if doc["is_parent"] == 0]
    
    if not child_docs:
        return []
        
    # --- A. ÇOKLU VEKTÖR ARAMASI (ANLAMSAL ARAMA) ---
    search_queries = expanded_queries if expanded_queries else [query]
    
    search_engine = VectorSearchEngine()
    query_vectors = [search_engine.generate_embedding(q) for q in search_queries]
    search_engine.close()
    
    # Çocuk parçalar için anlamsal skorları biriktir
    child_scores = {doc["id"]: 0.0 for doc in child_docs}
    
    for q_vector in query_vectors:
        for doc in child_docs:
            sim = cosine_similarity(q_vector, doc["embedding"])
            # Maksimum benzerlik skorunu sakla
            if sim > child_scores[doc["id"]]:
                child_scores[doc["id"]] = sim
                
    semantic_scored = []
    for doc in child_docs:
        doc_copy = doc.copy()
        doc_copy["score"] = child_scores[doc["id"]]
        semantic_scored.append(doc_copy)
        
    semantic_scored.sort(key=lambda x: x["score"], reverse=True)
    
    # --- B. KELİME (LEXICAL) ARAMA ---
    lexical_scores = compute_tfidf_scores(query, child_docs)
    lexical_scored = []
    for doc in child_docs:
        doc_copy = doc.copy()
        doc_copy["score"] = lexical_scores[doc["id"]]
        lexical_scored.append(doc_copy)
        
    lexical_scored.sort(key=lambda x: x["score"], reverse=True)
    
    # --- C. HİBRİT BİRLEŞTİRME (RRF) ---
    hybrid_results = reciprocal_rank_fusion(semantic_scored, lexical_scored)
    
    # --- D. EBEVEYN DÖKÜMANLARA ERİŞİM (PARENT RETRIEVAL) ---
    parent_results = []
    seen_parents = set()
    
    for child in hybrid_results:
        parent_id = child["parent_id"]
        if parent_id and parent_id not in seen_parents:
            parent_doc = get_document_by_id(parent_id)
            if parent_doc:
                # Orijinal alt parçanın skorunu ebeveyne yansıtıyoruz
                parent_doc["score"] = child["score"]
                parent_results.append(parent_doc)
                seen_parents.add(parent_id)
                
        # İstenen top_k adet ebeveyne ulaşıldığında döngüyü kes
        if len(parent_results) >= top_k:
            break
            
    return parent_results
