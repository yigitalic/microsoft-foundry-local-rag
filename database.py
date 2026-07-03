import sqlite3
import json

DB_FILE = "knowledge_base.db"

def init_db():
    """SQLite veritabanını ve gerekli tabloları hazırlar (V2.1 şeması ile)."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # 1. Dokümanlar tablosu (Özet ve etiket kolonları dahil)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT,
            embedding TEXT,
            file_type TEXT,
            source_file TEXT,
            summary TEXT,
            tags TEXT
        )
    """)
    
    # 2. Sohbet oturumları tablosu
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # 3. Sohbet mesajları tablosu
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
        )
    """)
    
    conn.commit()
    conn.close()
    print("[SİSTEM] Veritabanı V2.1 şemasıyla başlatıldı.")

# --- Doküman İşlemleri ---

def insert_document(title: str, content: str, embedding: list[float], file_type: str, source_file: str, summary: str = "", tags: str = ""):
    """Yeni bir dökümanı, onun embedding vektörünü ve metaverilerini veritabanına kaydeder."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    embedding_str = json.dumps(embedding)
    cursor.execute(
        "INSERT INTO documents (title, content, embedding, file_type, source_file, summary, tags) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (title, content, embedding_str, file_type, source_file, summary, tags)
    )
    conn.commit()
    conn.close()

def get_all_documents() -> list[dict]:
    """Veritabanındaki tüm dökümanları, vektörlerini ve metaverilerini getirir."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, content, embedding, file_type, source_file, summary, tags FROM documents")
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
            "source_file": row[5] if row[5] else row[1],
            "summary": row[6] if row[6] else "",
            "tags": row[7] if row[7] else ""
        })
    return docs

# --- Sohbet Geçmişi ve Oturum İşlemleri ---

def create_session(session_id: str, title: str):
    """Yeni bir sohbet oturumu oluşturur."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO chat_sessions (id, title) VALUES (?, ?)",
        (session_id, title)
    )
    conn.commit()
    conn.close()

def get_sessions() -> list[dict]:
    """Tüm sohbet oturumlarını tarihe göre azalan sırada getirir."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, created_at FROM chat_sessions ORDER BY created_at DESC")
    rows = cursor.fetchall()
    conn.close()
    
    sessions = []
    for row in rows:
        sessions.append({
            "id": row[0],
            "title": row[1],
            "created_at": row[2]
        })
    return sessions

def save_message(session_id: str, role: str, content: str):
    """Belirli bir oturuma mesaj kaydeder."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_messages (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content)
    )
    conn.commit()
    conn.close()

def get_session_messages(session_id: str) -> list[dict]:
    """Belirli bir oturumun tüm mesajlarını sırayla getirir."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM chat_messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = cursor.fetchall()
    conn.close()
    
    messages = []
    for row in rows:
        messages.append({
            "role": row[0],
            "content": row[1]
        })
    return messages

def delete_session(session_id: str):
    """Bir sohbet oturumunu ve ona bağlı tüm mesajları veritabanından siler."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    # Yabancı anahtar (Foreign Key) kısıtlaması nedeniyle cascade silme yapılacaktır
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()
