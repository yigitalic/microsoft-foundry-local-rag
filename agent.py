import asyncio
import json
import openai
from foundry_local_sdk import Configuration, FoundryLocalManager

# 1. Tanımlı Python araçları
def add_numbers(a: float, b: float) -> dict:
    print(f"   [SİSTEM] 'add_numbers' aracı çalıştırılıyor... Parametreler: a={a}, b={b}")
    return {"result": a + b}

def multiply_numbers(a: float, b: float) -> dict:
    print(f"   [SİSTEM] 'multiply_numbers' aracı çalıştırılıyor... Parametreler: a={a}, b={b}")
    return {"result": a * b}

available_functions = {
    "add_numbers": add_numbers,
    "multiply_numbers": multiply_numbers
}

# 2. Araç şemaları
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

async def main():
    print("Foundry Local Manager ve Web Service başlatılıyor...")
    
    # Yerel REST API servisini başlatmak için konfigürasyon
    web_config = Configuration.WebService(urls="http://127.0.0.1:0")
    config = Configuration(app_name="foundry-local-test", web=web_config)
    FoundryLocalManager.initialize(config)
    
    manager = FoundryLocalManager.instance
    catalog = manager.catalog
    
    manager.start_web_service()
    endpoint = manager.urls[0]
    print(f"Lokal API adresi: {endpoint}")
    
    print("phi-4-mini modeli belleğe yükleniyor...")
    model = catalog.get_model("phi-4-mini")
    model.load()
    print("Model hazır! Sohbet moduna geçiliyor.")
    
    client = openai.OpenAI(
        base_url=f"{endpoint}/v1" if not endpoint.endswith("/v1") else endpoint,
        api_key="not-needed"
    )
    
    # Sohbet geçmişi
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You must use the provided tools to perform calculations (addition and multiplication). Do not calculate them yourself. Call the tools."}
    ]
    
    print("\n=== INTERAKTIF LOCAL AI AGENT ===")
    print("Sohbeti başlatmak için yazın. Çıkmak için 'exit' veya 'quit' yazabilirsiniz.")
    
    # Programın interaktif çalışması için döngü (Terminalden girdi alabilmek için)
    # Staj sunumu ve demoları için mükemmel bir interaktif arayüzdür.
    while True:
        try:
            user_input = input("\nSiz: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("Sohbet kapatılıyor...")
                break
                
            if not user_input.strip():
                continue
                
            messages.append({"role": "user", "content": user_input})
            
            # Basit Yönlendirme (Heuristic Routing)
            # Matematiksel ifadeler varsa araç kullanımını zorunlu kılıyoruz.
            math_keywords = ["add", "multiply", "sum", "product", "calculate", "+", "*", "topla", "çarp", "hesapla"]
            is_math = any(keyword in user_input.lower() for keyword in math_keywords)
            
            # İlk adımda matematik sorusuysa tool çağrısını zorunlu yap, aksi halde otomatik bırak
            current_tool_choice = "required" if is_math else "auto"
            
            while True:
                response = client.chat.completions.create(
                    model=model.id,
                    messages=messages,
                    tools=tools,
                    tool_choice=current_tool_choice
                )
                
                choice = response.choices[0].message
                
                if choice.tool_calls:
                    messages.append(choice) # Modelin istek mesajını ekle
                    print(f"   [MODEL] {len(choice.tool_calls)} adet araç çağrısı talep etti.")
                    
                    for tool_call in choice.tool_calls:
                        func_name = tool_call.function.name
                        func_args = json.loads(tool_call.function.arguments)
                        
                        if func_name in available_functions:
                            # Aracı çalıştır
                            output = available_functions[func_name](**func_args)
                            
                            # Sonucu OpenAI formatında geçmişe ekle
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": json.dumps(output)
                            })
                            print(f"   [ARAÇ SONUCU] {func_name} -> {output}")
                        else:
                            print(f"   [HATA] {func_name} tanımlı bir fonksiyon değil.")
                            
                    # Sonraki adımda serbest bırak (auto)
                    current_tool_choice = "auto"
                else:
                    # Model nihai yanıtı ürettiğinde ekrana yazdır
                    print(f"\nYapay Zeka: {choice.content}")
                    messages.append({"role": "assistant", "content": choice.content})
                    break
                    
        except KeyboardInterrupt:
            print("\nSohbet sonlandırılıyor...")
            break
        except Exception as e:
            print(f"Bir hata oluştu: {e}")
            break
            
    # Temizlik
    print("\nTemizlik yapılıyor...")
    model.unload()
    manager.stop_web_service()
    print("Süreç başarıyla sonlandırıldı.")

if __name__ == "__main__":
    asyncio.run(main())
