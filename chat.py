import asyncio
import time
from foundry_local_sdk import Configuration, FoundryLocalManager

async def main():
    print("Foundry Local Manager başlatılıyor...")
    try:
        # Konfigürasyonu ayarla
        config = Configuration(app_name="foundry-local-test")
        FoundryLocalManager.initialize(config)
        
        manager = FoundryLocalManager.instance
        catalog = manager.catalog
        
        # Hafif ve hızlı inen bir kodlama modeli seçiyoruz (qwen2.5-coder-0.5b)
        model_name = "qwen2.5-coder-0.5b"
        print(f"\nModel alınıyor: {model_name}")
        model = catalog.get_model(model_name)
        
        # Model zaten önbellekte mi kontrol et
        if not model.is_cached:
            print(f"Model yerel cihazda bulunamadı. İndirme başlatılıyor (Yaklaşık 300-400 MB)...")
            start_time = time.time()
            model.download()
            print(f"İndirme tamamlandı! Süre: {time.time() - start_time:.2f} saniye")
        else:
            print("Model zaten yerel olarak önbellekte (cached) mevcut.")
            
        print("\nModel belleğe (RAM/VRAM) yükleniyor...")
        start_time = time.time()
        model.load()
        print(f"Yükleme tamamlandı! Süre: {time.time() - start_time:.2f} saniye")
        
        print("\nChat istemcisi oluşturuluyor...")
        chat_client = model.get_chat_client()
        
        # Test sorusu
        prompt = "Explain Binary Search algorithm in one sentence."
        print(f"\nSoru: '{prompt}'")
        print("\nYapay zeka yanıt üretiyor...")
        
        messages = [{"role": "user", "content": prompt}]
        response = chat_client.complete_chat(messages=messages)
        
        # Yanıtı ekrana yazdır
        print("\n--- Model Yanıtı ---")
        print(response.choices[0].message.content)
        print("--------------------")
        
        # Belleği boşaltmak için modeli unload et
        print("\nModel bellekten kaldırılıyor...")
        model.unload()
        print("Bellek temizlendi.")
        
    except Exception as e:
        print(f"Hata oluştu: {e}")

if __name__ == "__main__":
    asyncio.run(main())
