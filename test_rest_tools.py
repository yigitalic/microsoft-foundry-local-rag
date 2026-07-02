import asyncio
import json
import openai
from foundry_local_sdk import Configuration, FoundryLocalManager

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
    }
]

async def main():
    print("Foundry Local Manager ve Web Service başlatılıyor...")
    
    # WebService konfigürasyonunu ekliyoruz
    web_config = Configuration.WebService(urls="http://127.0.0.1:0")
    config = Configuration(app_name="foundry-local-test", web=web_config)
    FoundryLocalManager.initialize(config)
    
    manager = FoundryLocalManager.instance
    catalog = manager.catalog
    
    # Web servisini başlat
    manager.start_web_service()
    endpoint = manager.urls[0]
    print(f"Web Service endpoint adresi: {endpoint}")
    
    # Model yükle
    model = catalog.get_model("phi-4-mini")
    model.load()
    
    # Standart OpenAI istemcisini bağla
    client = openai.OpenAI(
        base_url=f"{endpoint}/v1" if not endpoint.endswith("/v1") else endpoint,
        api_key="not-needed"
    )
    
    prompt = "Please calculate 125.5 + 74.5."
    print(f"\nSoru: '{prompt}'")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You must use the provided tools for any calculations. Do not compute answers yourself."},
        {"role": "user", "content": prompt}
    ]
    
    print("REST API üzerinden istek gönderiliyor...")
    try:
        response = client.chat.completions.create(
            model=model.id,
            messages=messages,
            tools=tools,
            tool_choice="required"
        )
        choice = response.choices[0].message
        print("\n--- REST API Model Çıktısı ---")
        print(f"Content: {choice.content}")
        print(f"Tool Calls: {choice.tool_calls}")
        print("-------------------------------")
    except Exception as e:
        print(f"İstek sırasında hata: {e}")
        
    # Temizlik
    model.unload()
    manager.stop_web_service()
    print("Web servisi ve model kapatıldı.")

if __name__ == "__main__":
    asyncio.run(main())
