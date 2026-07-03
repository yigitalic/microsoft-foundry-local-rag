import sqlite3
import json

DB_FILE = "knowledge_base.db"

def init_db():
    """SQLite veritabanını ve gerekli tabloları hazırlar."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT,
            embedding TEXT
        )
    """)
    conn.commit()
    conn.close()
    print("[SİSTEM] Veritabanı başlatıldı ve tablo oluşturuldu.")

def insert_document(title: str, content: str, embedding: list[float]):
    """Yeni bir dökümanı ve onun embedding vektörünü veritabanına kaydeder."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Vektörü JSON string olarak kaydediyoruz
    embedding_str = json.dumps(embedding)
    cursor.execute(
        "INSERT INTO documents (title, content, embedding) VALUES (?, ?, ?)",
        (title, content, embedding_str)
    )
    conn.commit()
    conn.close()

def get_all_documents() -> list[dict]:
    """Veritabanındaki tüm dökümanları ve vektörlerini getirir."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, content, embedding FROM documents")
    rows = cursor.fetchall()
    conn.close()
    
    docs = []
    for row in rows:
        docs.append({
            "id": row[0],
            "title": row[1],
            "content": row[2],
            "embedding": json.loads(row[3])
        })
    return docs
