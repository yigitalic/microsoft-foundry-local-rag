import math
from foundry_local_sdk import Configuration, FoundryLocalManager

def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """İki vektör arasındaki kosinüs benzerliğini hesaplar."""
    dot_product = sum(x * y for x, y in zip(v1, v2))
    norm_v1 = math.sqrt(sum(x * x for x in v1))
    norm_v2 = math.sqrt(sum(y * y for y in v2))
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

class VectorSearchEngine:
    """Yerel model ile embedding üreten ve kosinüs benzerliği araması yapan motor."""
    
    def __init__(self):
        # Önbelleği ortak klasörde tutacak şekilde başlatıyoruz
        config = Configuration(app_name="foundry-local-test")
        try:
            FoundryLocalManager.initialize(config)
        except Exception:
            # Singleton zaten başlatılmışsa
            pass
            
        self.manager = FoundryLocalManager.instance
        self.catalog = self.manager.catalog
        
        # 0.6B parametreli hafif embedding modelini alıyoruz
        model_name = "qwen3-embedding-0.6b"
        self.model = self.catalog.get_model(model_name)
        
        if not self.model.is_cached:
            print(f"[SİSTEM] Embedding modeli yerel cihazda yok. İndiriliyor (Model: {model_name})...")
            self.model.download()
            
        print("[SİSTEM] Embedding modeli belleğe yükleniyor...")
        self.model.load()
        self.client = self.model.get_embedding_client()
        print("[SİSTEM] Embedding istemcisi hazır.")

    def generate_embedding(self, text: str) -> list[float]:
        """Metnin sayısal embedding vektörünü (list[float]) üretir."""
        if not text.strip():
            raise ValueError("Boş metin için embedding üretilemez.")
        response = self.client.generate_embedding(text)
        return response.data[0].embedding

    def close(self):
        """Modeli bellekten kaldırır."""
        print("[SİSTEM] Embedding modeli bellekten çıkarılıyor...")
        self.model.unload()
        print("[SİSTEM] Bellek temizlendi.")
