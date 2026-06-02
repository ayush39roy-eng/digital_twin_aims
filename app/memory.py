import sqlite3
from datetime import datetime
from app.config import SQLITE_PATH


class ShortTermMemory:
    def __init__(self):
        self._history: list[dict] = []

    def add_turn(self, role: str, content: str):
        self._history.append({"role": role, "content": content})

    def get_history(self) -> list[dict]:
        return self._history[-12:]

    def clear(self):
        self._history = []


class LongTermMemory:
    def __init__(self):
        self._init_db()

    def _conn(self):
        return sqlite3.connect(SQLITE_PATH)

    def _init_db(self):
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS long_term_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    content TEXT NOT NULL
                )"""
            )
            conn.commit()

    def save(self, session_id: str, content: str):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO long_term_memory (session_id, timestamp, content) VALUES (?, ?, ?)",
                (session_id, datetime.utcnow().isoformat(), content),
            )
            conn.commit()

    def load(self, session_id: str) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT content FROM long_term_memory WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [row[0] for row in rows]

    def summarize_and_save(self, session_id: str, conversation_history: list[dict], gemini_client):
        if not conversation_history:
            return
        history_text = "\n".join(
            f"{t['role'].capitalize()}: {t['content']}" for t in conversation_history
        )
        prompt = (
            "Read this conversation and extract 3-5 concise facts worth remembering about "
            "what the user is interested in, curious about, or working on. "
            "Return only a numbered list, nothing else.\n\n"
            f"{history_text}"
        )
        summary = gemini_client(prompt)
        self.save(session_id, summary)
