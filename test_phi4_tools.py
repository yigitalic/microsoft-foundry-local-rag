import asyncio
import json
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
    print("Foundry Local Manager başlatılıyor...")
    config = Configuration(app_name="foundry-local-test")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    catalog = manager.catalog
    
    model = catalog.get_model("phi-4-mini")
    model.load()
    chat_client = model.get_chat_client()
    
    # Soru
    prompt = "What is 125.5 + 74.5?"
    print(f"\nSoru: '{prompt}'")
    
    messages = [{"role": "user", "content": prompt}]
    
    response = chat_client.complete_chat(messages=messages, tools=tools)
    choice = response.choices[0].message
    
    print("\n--- Model Çıktısı ---")
    print(f"Content: {choice.content}")
    print(f"Tool Calls: {choice.tool_calls}")
    print("----------------------")
    
    model.unload()

if __name__ == "__main__":
    asyncio.run(main())
