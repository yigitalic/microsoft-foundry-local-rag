import sqlite3
import json

DB_FILE = "knowledge_base.db"

def init_db():
    """SQLite veritabanını ve gerekli tabloları hazırlar (V3.0 Parent-Child şeması ile)."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # 1. Dokümanlar tablosu (Parent-Child ilişkisi dahil)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            content TEXT,
            embedding TEXT,
            file_type TEXT,
            source_file TEXT,
            summary TEXT,
            tags TEXT,
            parent_id INTEGER,
            is_parent INTEGER DEFAULT 0
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
    print("[SİSTEM] Veritabanı V3.0 şemasıyla başlatıldı.")

# --- Doküman İşlemleri ---

def insert_document(title: str, content: str, embedding: list[float] = None, file_type: str = "text", 
                    source_file: str = "", summary: str = "", tags: str = "", 
                    parent_id: int = None, is_parent: int = 0) -> int:
    """Yeni bir dökümanı (Parent veya Child) veritabanına kaydeder ve onun ID'sini (rowid) döner."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Ebeveyn parçaların embedding'i olmaz, sadece alt (child) parçaların olur
    embedding_str = json.dumps(embedding) if embedding is not None else None
    
    cursor.execute(
        """INSERT INTO documents 
           (title, content, embedding, file_type, source_file, summary, tags, parent_id, is_parent) 
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (title, content, embedding_str, file_type, source_file, summary, tags, parent_id, is_parent)
    )
    last_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return last_id

def get_all_documents() -> list[dict]:
    """Veritabanındaki tüm dökümanları (sadece ebeveyn olmayanları veya tüm listeyi) getirir."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, content, embedding, file_type, source_file, summary, tags, parent_id, is_parent FROM documents")
    rows = cursor.fetchall()
    conn.close()
    
    docs = []
    for row in rows:
        docs.append({
            "id": row[0],
            "title": row[1],
            "content": row[2],
            "embedding": json.loads(row[3]) if row[3] else None,
            "file_type": row[4] if row[4] else "text",
            "source_file": row[5] if row[5] else row[1],
            "summary": row[6] if row[6] else "",
            "tags": row[7] if row[7] else "",
            "parent_id": row[8],
            "is_parent": row[9]
        })
    return docs

def get_document_by_id(doc_id: int) -> dict:
    """Belirli bir dökümanı ID değerine göre getirir (Ebeveyn dökümanı bulmak için)."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, content, file_type, source_file, summary, tags FROM documents WHERE id = ?", (doc_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "id": row[0],
            "title": row[1],
            "content": row[2],
            "file_type": row[3],
            "source_file": row[4],
            "summary": row[5],
            "tags": row[6]
        }
    return None

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
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()
