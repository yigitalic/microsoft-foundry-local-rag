import asyncio
import json
import time
from foundry_local_sdk import Configuration, FoundryLocalManager

# 1. Tanımlayacağımız araçların gerçek Python fonksiyonları
def add_numbers(a: float, b: float) -> dict:
    print(f"-> [ARAÇ ÇALIŞTIRILIYOR] add_numbers(a={a}, b={b})")
    return {"result": a + b}

def multiply_numbers(a: float, b: float) -> dict:
    print(f"-> [ARAÇ ÇALIŞTIRILIYOR] multiply_numbers(a={a}, b={b})")
    return {"result": a * b}

# 2. Araçların modelin anlayacağı OpenAI formatındaki şemaları
tools = [
    {
        "type": "function",
        "function": {
            "name": "add_numbers",
            "description": "Adds two numbers together.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "The first number."},
                    "b": {"type": "number", "description": "The second number."}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "multiply_numbers",
            "description": "Multiplies two numbers together.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "The first number."},
                    "b": {"type": "number", "description": "The second number."}
                },
                "required": ["a", "b"]
            }
        }
    }
]

# Araç isimlerini Python fonksiyonlarıyla eşleştiriyoruz
available_functions = {
    "add_numbers": add_numbers,
    "multiply_numbers": multiply_numbers
}

async def main():
    print("Foundry Local Manager başlatılıyor...")
    config = Configuration(app_name="foundry-local-test")
    FoundryLocalManager.initialize(config)
    
    manager = FoundryLocalManager.instance
    catalog = manager.catalog
    
    # Microsoft'un phi-4-mini modelini seçiyoruz
    model_name = "phi-4-mini"
    model = catalog.get_model(model_name)
    
    # Model zaten önbellekte mi kontrol et, değilse indir
    if not model.is_cached:
        print(f"Model yerel cihazda bulunamadı. İndirme başlatılıyor (Yaklaşık 2.2 GB)...")
        start_time = time.time()
        model.download()
        print(f"İndirme tamamlandı! Süre: {time.time() - start_time:.2f} saniye")
    
    print("Model belleğe yükleniyor...")
    model.load()
    chat_client = model.get_chat_client()
    print("Hazır!")
    
    # Modelin araçları tetiklemesi için tasarlanmış bir soru soruyoruz
    prompt = "Can you add 125.5 and 74.5, and then multiply the result by 3?"
    print(f"\nSoru: '{prompt}'")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You MUST use the provided tools to perform calculations (addition and multiplication). Do not calculate them yourself. Call the tools."},
        {"role": "user", "content": prompt}
    ]
    
    # İlk sohbet çağrısı (Araçlarla birlikte)
    print("\nModel düşünülüyor ve araç çağrısı değerlendiriliyor...")
    response = chat_client.complete_chat(messages=messages, tools=tools)
    choice = response.choices[0].message
    
    # Eğer model araç çağrısı yapmak istediyse döngüye giriyoruz
    while choice.tool_calls:
        print(f"\n[MODEL] Araç çağrısı talep etti ({len(choice.tool_calls)} adet):")
        messages.append(choice)  # Modelin cevabını geçmişe ekle
        
        for tool_call in choice.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Çağrılacak fonksiyonu bul ve çalıştır
            if function_name in available_functions:
                func = available_functions[function_name]
                tool_output = func(**function_args)
                
                # Fonksiyon sonucunu OpenAI standardına göre geçmişe ekle
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_output)
                })
                print(f"-> [ARAÇ SONUCU] {tool_output}")
            else:
                print(f"Hata: {function_name} tanımlı bir fonksiyon değil.")
        
        # Sonuçları modele geri göndererek bir sonraki adımı alıyoruz
        print("\nSonuçlar modele gönderiliyor, model yanıt hazırlıyor...")
        response = chat_client.complete_chat(messages=messages, tools=tools)
        choice = response.choices[0].message
        
    print("\n--- Nihai Model Yanıtı ---")
    print(choice.content)
    print("--------------------------")
    
    # Belleği temizle
    model.unload()
    print("\nBellek temizlendi.")

if __name__ == "__main__":
    asyncio.run(main())
