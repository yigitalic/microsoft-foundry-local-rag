import asyncio
from foundry_local_sdk import Configuration, FoundryLocalManager

async def main():
    print("Foundry Local Manager başlatılıyor...")
    try:
        # Konfigürasyonu ayarla
        config = Configuration(app_name="foundry-local-test")
        FoundryLocalManager.initialize(config)
        
        manager = FoundryLocalManager.instance
        catalog = manager.catalog
        
        # Donanımınıza uygun olan modelleri katalogdan listele
        print("\nMevcut Yerel Yapay Zeka Modelleri:")
        models = catalog.list_models()
        if not models:
            print("Katalogda uygun model bulunamadı.")
            return
            
        print(f"Toplam {len(models)} model bulundu.")
        for model in models:
            status = "İndirilmiş (Cached)" if model.is_cached else "İndirilmemiş"
            print(f" - {model.alias} (ID: {model.id}, Durum: {status}, Bağlam Boyutu: {model.context_length})")
            
    except Exception as e:
        print(f"Hata oluştu: {e}")

if __name__ == "__main__":
    asyncio.run(main())
