import asyncio
import openai
from database import init_db
from retrieval import retrieve_context
from foundry_local_sdk import Configuration, FoundryLocalManager

async def run_evaluation():
    print("=== YEREL RAG DEĞERLENDİRME (EVALUATOR) BAŞLATILDI ===")
    
    # Veritabanını ilklendir
    init_db()
    
    # Test senaryoları
    test_cases = [
        {
            "category": "Bilgi Tabanı Sorusu",
            "question": "Final staj raporunun son teslim tarihi ve saati nedir?"
        },
        {
            "category": "Kod / Teknik Soru",
            "question": "Microsoft Foundry Local hangi donanımları otomatik algılar ve hızlandırır?"
        },
        {
            "category": "Sınır Durum (Bilinmeyen Bilgi)",
            "question": "Yaz okulu stajyerlerine servis veya ulaşım imkanı sağlanıyor mu?"
        }
    ]
    
    # 1. Local LLM servisini başlat
    print("\n[EVALUATOR] Yerel Phi-4-mini modeli ve Web servisi yükleniyor...")
    web_config = Configuration.WebService(urls="http://127.0.0.1:0")
    config = Configuration(app_name="foundry-local-test", web=web_config)
    try:
        FoundryLocalManager.initialize(config)
    except Exception:
        pass
    manager = FoundryLocalManager.instance
    manager.start_web_service()
    endpoint = manager.urls[0]
    
    model = manager.catalog.get_model("phi-4-mini")
    model.load()
    
    client = openai.OpenAI(base_url=f"{endpoint}/v1", api_key="not-needed")
    
    print("\n" + "="*60)
    
    for i, case in enumerate(test_cases, 1):
        question = case["question"]
        print(f"\nTEST {i}: [{case['category']}]")
        print(f"SORU: '{question}'")
        
        # A. Doküman Arama (Retrieval)
        chunks = retrieve_context(question, top_k=2)
        context_text = ""
        sources = []
        for idx, chunk in enumerate(chunks, 1):
            context_text += f"\n[Parça {idx}] (Kaynak: {chunk['title']})\n{chunk['content']}\n"
            sources.append(chunk['title'])
            
        print(f"-> Arama Bitti. Bulunan Kaynaklar: {', '.join(sources)}")
        
        # B. Cevap Üretimi (RAG Generation)
        system_prompt = (
            "You are a highly analytical on-device AI agent. Your task is to answer the user's question based ONLY on the provided Context.\n"
            "If the Context does not contain the answer, reply with 'I cannot find this information in my database.'\n"
            "Do NOT make up facts. Never use external knowledge. Always keep your response grounded."
        )
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}"
        
        response = client.chat.completions.create(
            model=model.id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0
        )
        answer = response.choices[0].message.content.strip()
        print(f"CEVAP:\n{answer}")
        
        # C. LLM-as-a-Judge Değerlendirmesi
        judge_prompt = f"""You are an expert AI quality inspector judging a RAG system's response.
Evaluate the response quality based ONLY on the provided Question, Retrieved Context, and Generated Answer.

Question:
{question}

Retrieved Context:
{context_text}

Generated Answer:
{answer}

Rate the answer on these two metrics:
1. Faithfulness (Groundedness): Score from 0 to 5. 5 means the answer is 100% supported by the context with zero hallucinations. 0 means the answer contains fabricated facts not in the context.
2. Relevance: Score from 0 to 5. 5 means the answer directly and completely answers the question.

You MUST respond ONLY in the following format (no other text, no intro, no markdown codeblock):
Faithfulness: [Score]/5
Relevance: [Score]/5
Explanation: [1-sentence reason for scores]"""

        judge_response = client.chat.completions.create(
            model=model.id,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0
        )
        
        evaluation = judge_response.choices[0].message.content.strip()
        print(f"\n--- YAPAY ZEKA DEĞERLENDİRME ANALİZİ ---")
        print(evaluation)
        print("="*60)
        
    # Temizlik
    model.unload()
    manager.stop_web_service()
    print("\nDeğerlendirme başarıyla tamamlandı.")

def run_evaluation_for_ui(client, model_id: str) -> list[dict]:
    """Streamlit arayüzü için değerlendirme senaryolarını koşturur ve sonuçları döner."""
    test_cases = [
        {
            "category": "Bilgi Tabanı Sorusu",
            "question": "Final staj raporunun son teslim tarihi ve saati nedir?"
        },
        {
            "category": "Kod / Teknik Soru",
            "question": "Microsoft Foundry Local hangi donanımları otomatik algılar ve hızlandırır?"
        },
        {
            "category": "Sınır Durum (Bilinmeyen Bilgi)",
            "question": "Yaz okulu stajyerlerine servis veya ulaşım imkanı sağlanıyor mu?"
        }
    ]
    
    results = []
    
    for case in test_cases:
        question = case["question"]
        
        # A. Doküman Arama (Retrieval)
        chunks = retrieve_context(question, top_k=2)
        context_text = ""
        sources = []
        for idx, chunk in enumerate(chunks, 1):
            context_text += f"\n[Parça {idx}] (Kaynak: {chunk['title']})\n{chunk['content']}\n"
            sources.append(chunk['title'])
            
        # B. Cevap Üretimi (RAG Generation)
        system_prompt = (
            "You are a highly analytical on-device AI agent. Your task is to answer the user's question based ONLY on the provided Context.\n"
            "If the Context does not contain the answer, reply with 'I cannot find this information in my database.'\n"
            "Do NOT make up facts. Never use external knowledge. Always keep your response grounded."
        )
        user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}"
        
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            answer = response.choices[0].message.content.strip()
        except Exception as e:
            answer = f"Error generating answer: {e}"
        
        # C. LLM-as-a-Judge Değerlendirmesi
        judge_prompt = f"""You are an expert AI quality inspector judging a RAG system's response.
Evaluate the response quality based ONLY on the provided Question, Retrieved Context, and Generated Answer.

Question:
{question}

Retrieved Context:
{context_text}

Generated Answer:
{answer}

Rate the answer on these two metrics:
1. Faithfulness (Groundedness): Score from 0 to 5. 5 means the answer is 100% supported by the context with zero hallucinations. 0 means the answer contains fabricated facts not in the context.
2. Relevance: Score from 0 to 5. 5 means the answer directly and completely answers the question.

You MUST respond ONLY in the following format (no other text, no intro, no markdown codeblock):
Faithfulness: [Score]/5
Relevance: [Score]/5
Explanation: [1-sentence reason for scores]"""

        faithfulness = 0
        relevance = 0
        explanation = "Değerlendirme başarısız."
        
        try:
            judge_response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0
            )
            evaluation = judge_response.choices[0].message.content.strip()
            
            for line in evaluation.split("\n"):
                if line.lower().startswith("faithfulness:"):
                    parts = line.split(":", 1)[1].strip().split("/")
                    faithfulness = int(parts[0]) if parts else 0
                elif line.lower().startswith("relevance:"):
                    parts = line.split(":", 1)[1].strip().split("/")
                    relevance = int(parts[0]) if parts else 0
                elif line.lower().startswith("explanation:"):
                    explanation = line.split(":", 1)[1].strip()
        except Exception as e:
            explanation = f"Error in evaluation: {e}"
            
        results.append({
            "category": case["category"],
            "question": question,
            "context": context_text,
            "sources": sources,
            "answer": answer,
            "faithfulness": faithfulness,
            "relevance": relevance,
            "explanation": explanation
        })
        
    return results

if __name__ == "__main__":
    asyncio.run(run_evaluation())

