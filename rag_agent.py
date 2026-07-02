import asyncio
import openai
from foundry_local_sdk import Configuration, FoundryLocalManager
from retrieval import retrieve_context

async def generate_rag_answer(query: str) -> str:
    # 1. Bilgi tabanından en alakalı doküman parçalarını getir
    print("\n[RAG] Bilgi tabanında arama yapılıyor...")
    context_chunks = retrieve_context(query, top_k=2)
    
    if not context_chunks:
        return "Veritabanında sorguyla alakalı doküman bulunamadı."
        
    print(f"-> {len(context_chunks)} adet alakalı parça bulundu. Metin birleştiriliyor...")
    
    # Parçaları ve kaynaklarını birleştir
    context_text = ""
    sources = set()
    for idx, chunk in enumerate(context_chunks, 1):
        context_text += f"\n[Doküman {idx}] (Kaynak: {chunk['title']})\n{chunk['content']}\n"
        # Orijinal dosya adını al (örn: staj_rehberi.txt - Parça 1 -> staj_rehberi.txt)
        source_name = chunk['title'].split(" - ")[0]
        sources.add(source_name)
        
    # 2. Local LLM (phi-4-mini) servisini başlat
    print("\n[RAG] Local LLM (phi-4-mini) başlatılıyor...")
    web_config = Configuration.WebService(urls="http://127.0.0.1:0")
    config = Configuration(app_name="foundry-local-test", web=web_config)
    
    try:
        FoundryLocalManager.initialize(config)
    except Exception:
        # Singleton zaten başlatılmışsa
        pass
        
    manager = FoundryLocalManager.instance
    catalog = manager.catalog
    
    manager.start_web_service()
    endpoint = manager.urls[0]
    
    model = catalog.get_model("phi-4-mini")
    model.load()
    
    # 3. OpenAI istemcisi ile RAG promptunu gönder
    client = openai.OpenAI(
        base_url=f"{endpoint}/v1" if not endpoint.endswith("/v1") else endpoint,
        api_key="not-needed"
    )
    
    system_prompt = (
        "You are a helpful local assistant. Answer the user's question using ONLY the provided Context below. "
        "If the Context does not contain the answer, reply with 'I cannot find this information in my database.' "
        "Do not make up facts or use external knowledge. Always keep your response grounded."
    )
    
    user_prompt = f"Context:\n{context_text}\n\nQuestion: {query}"
    
    print("[RAG] Yapay zeka yerel yanıtı üretiyor...")
    response = client.chat.completions.create(
        model=model.id,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0 # Daha kararlı ve uydurmasız yanıtlar için
    )
    
    raw_answer = response.choices[0].message.content
    
    # Kaynakları cevabın sonuna ekle
    source_citation = "\n\nSources: " + ", ".join(sources)
    final_answer = raw_answer + source_citation
    
    # 4. Model ve servis kapatma temizliği (Bellek koruması)
    model.unload()
    manager.stop_web_service()
    
    return final_answer

async def main():
    print("=== LOCAL RAG AGENT TEST PANELİ ===")
    
    # Örnek Sorular
    questions = [
        "Mentor Barbaros Bey ile haftalık toplantı ne zaman yapılacak?",
        "Final staj raporunun son teslim tarihi ve saati nedir?",
        "Microsoft Foundry Local hangi donanım hızlandırıcıları destekler?"
    ]
    
    for q in questions:
        print("\n" + "="*50)
        print(f"SORU: {q}")
        answer = await generate_rag_answer(q)
        print(f"\nCEVAP:\n{answer}")
        
if __name__ == "__main__":
    asyncio.run(main())
