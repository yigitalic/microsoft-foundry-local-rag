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
    web_config = Configuration.WebService(urls="http://127.0.0.1:0")
    config = Configuration(app_name="foundry-local-test", web=web_config)
    FoundryLocalManager.initialize(config)
    
    manager = FoundryLocalManager.instance
    catalog = manager.catalog
    
    manager.start_web_service()
    endpoint = manager.urls[0]
    
    model = catalog.get_model("phi-4-mini")
    model.load()
    
    client = openai.OpenAI(
        base_url=f"{endpoint}/v1" if not endpoint.endswith("/v1") else endpoint,
        api_key="not-needed"
    )
    
    prompt = "Can you add 125.5 and 74.5, and then multiply the result by 3?"
    print(f"\nSoru: '{prompt}'")
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. You must use the provided tools for any calculations. Do not compute answers yourself."},
        {"role": "user", "content": prompt}
    ]
    
    # 1. Turn: force tool choice because it's the first step
    print("\n--- TURN 1 (tool_choice='required') ---")
    response = client.chat.completions.create(
        model=model.id,
        messages=messages,
        tools=tools,
        tool_choice="required"
    )
    choice = response.choices[0].message
    print(f"Content: {choice.content}")
    print(f"Tool Calls: {choice.tool_calls}")
    
    if choice.tool_calls:
        messages.append(choice)
        
        # Execute tool calls
        for tool_call in choice.tool_calls:
            print(f"Executing: {tool_call.function.name} with {tool_call.function.arguments}")
            if tool_call.function.name == "add_numbers":
                args = json.loads(tool_call.function.arguments)
                res = {"result": args["a"] + args["b"]}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(res)
                })
                print(f"Result appended: {res}")
                
        # 2. Turn: try tool_choice="auto" now that it has context
        print("\n--- TURN 2 (tool_choice='auto') ---")
        response2 = client.chat.completions.create(
            model=model.id,
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        choice2 = response2.choices[0].message
        print(f"Content: {choice2.content}")
        print(f"Tool Calls: {choice2.tool_calls}")
        
        if choice2.tool_calls:
            messages.append(choice2)
            # Execute tool call
            for tool_call in choice2.tool_calls:
                print(f"Executing: {tool_call.function.name} with {tool_call.function.arguments}")
                if tool_call.function.name == "multiply_numbers":
                    args = json.loads(tool_call.function.arguments)
                    res = {"result": args["a"] * args["b"]}
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(res)
                    })
                    print(f"Result appended: {res}")
                    
            # 3. Turn: final response
            print("\n--- TURN 3 (tool_choice='auto') ---")
            response3 = client.chat.completions.create(
                model=model.id,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
            choice3 = response3.choices[0].message
            print(f"Content: {choice3.content}")
            print(f"Tool Calls: {choice3.tool_calls}")

    model.unload()
    manager.stop_web_service()

if __name__ == "__main__":
    asyncio.run(main())
