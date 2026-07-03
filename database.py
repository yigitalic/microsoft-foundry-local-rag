import sqlite3
import json

DB_FILE = "knowledge_base.db"

def init_db():
    """SQLite veritabanını ve gerekli tabloları hazırlar (V2 şeması ile)."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT,
            embedding TEXT,
            file_type TEXT,
            source_file TEXT
        )
    """)
    conn.commit()
    conn.close()
    print("[SİSTEM] Veritabanı V2 şemasıyla başlatıldı.")

def insert_document(title: str, content: str, embedding: list[float], file_type: str, source_file: str):
    """Yeni bir dökümanı, onun embedding vektörünü ve metaverilerini veritabanına kaydeder."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    embedding_str = json.dumps(embedding)
    cursor.execute(
        "INSERT INTO documents (title, content, embedding, file_type, source_file) VALUES (?, ?, ?, ?, ?)",
        (title, content, embedding_str, file_type, source_file)
    )
    conn.commit()
    conn.close()

def get_all_documents() -> list[dict]:
    """Veritabanındaki tüm dökümanları, vektörlerini ve metaverilerini getirir."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, content, embedding, file_type, source_file FROM documents")
    rows = cursor.fetchall()
    conn.close()
    
    docs = []
    for row in rows:
        docs.append({
            "id": row[0],
            "title": row[1],
            "content": row[2],
            "embedding": json.loads(row[3]),
            "file_type": row[4] if row[4] else "txt",
            "source_file": row[5] if row[5] else row[1]
        })
    return docs
