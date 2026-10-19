"""Persistent long-term memory backed by SQLite.

This is application data that outlives a single graph run (and a process
restart). It is different from:
  - graph state:   lives only for one run of the graph
  - checkpointing: LangGraph's snapshots of graph state, used for resume/replay

Retrieval is deliberately simple: keyword overlap, no embeddings.
"""

import re
import sqlite3
from datetime import datetime, timezone

_STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "for", "is", "are", "be",
    "i", "should", "do", "does", "what", "how", "before", "after", "with", "it",
    "my", "me", "can", "learn", "about", "this", "that", "you", "your",
}


# Matches the summaries written by the save_memory node.
_SUMMARY_RE = re.compile(r"^Previously researched '(.+?)': ", re.DOTALL)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def topic_key(question: str) -> str:
    """Normalise a question so trivial variations map to the same memory row."""
    return re.sub(r"\s+", " ", question).strip().rstrip("?!. ").lower()


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in _STOPWORDS}


class MemoryStore:
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS facts (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   category TEXT NOT NULL,        -- 'preference' or 'fact'
                   content TEXT NOT NULL UNIQUE,
                   source TEXT NOT NULL,
                   created_at TEXT NOT NULL)"""
        )
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        """Add the `topic` column to older databases and collapse duplicate facts.

        topic = normalised question. A UNIQUE index on it means one fact per question.
        Preferences have topic NULL (SQLite allows many NULLs in a UNIQUE index).
        """
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(facts)")}
        if "topic" not in cols:
            self.conn.execute("ALTER TABLE facts ADD COLUMN topic TEXT")

        # Backfill topics for facts saved before this column existed; keep the newest per topic.
        rows = self.conn.execute(
            "SELECT id, content FROM facts WHERE category = 'fact' AND topic IS NULL ORDER BY id"
        ).fetchall()
        newest: dict[str, int] = {}
        for row_id, content in rows:
            m = _SUMMARY_RE.match(content)
            if m:
                newest[topic_key(m.group(1))] = row_id  # later ids overwrite earlier ones
        keep = set(newest.values())
        for row_id, content in rows:
            m = _SUMMARY_RE.match(content)
            if m and row_id not in keep:
                self.conn.execute("DELETE FROM facts WHERE id = ?", (row_id,))
        for key, row_id in newest.items():
            # An already-topic'd row for the same question wins over the legacy one.
            if self.conn.execute("SELECT 1 FROM facts WHERE topic = ?", (key,)).fetchone():
                self.conn.execute("DELETE FROM facts WHERE id = ?", (row_id,))
            else:
                self.conn.execute("UPDATE facts SET topic = ? WHERE id = ?", (key, row_id))

        self.conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_topic ON facts(topic)")

    def save(self, content: str, category: str = "fact", source: str = "user") -> bool:
        """Insert a memory. Returns False if the exact content already exists."""
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO facts (category, content, source, created_at) VALUES (?, ?, ?, ?)",
            (category, content.strip(), source, _now()),
        )
        self.conn.commit()
        return cur.rowcount == 1

    def upsert_fact(self, topic: str, content: str, source: str) -> str:
        """Store the fact for a topic, replacing any older fact for the same topic.

        Returns "added" or "updated".
        """
        key = topic_key(topic)
        existed = self.conn.execute("SELECT 1 FROM facts WHERE topic = ?", (key,)).fetchone()
        self.conn.execute(
            """INSERT INTO facts (category, content, source, created_at, topic)
               VALUES ('fact', ?, ?, ?, ?)
               ON CONFLICT(topic) DO UPDATE SET
                   content = excluded.content, source = excluded.source,
                   created_at = excluded.created_at""",
            (content.strip(), source, _now(), key),
        )
        self.conn.commit()
        return "updated" if existed else "added"

    def all(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, category, content, source, created_at FROM facts ORDER BY id"
        ).fetchall()
        return [dict(zip(("id", "category", "content", "source", "created_at"), r)) for r in rows]

    def relevant(self, question: str, limit: int = 5) -> list[str]:
        """Preferences always apply; facts are ranked by keyword overlap with the question."""
        q = _keywords(question)
        prefs, scored = [], []
        for m in self.all():
            if m["category"] == "preference":
                prefs.append(f"User preference: {m['content']}")
            else:
                score = len(q & _keywords(m["content"]))
                if score:
                    scored.append((score, m["id"], m["content"]))
        scored.sort(key=lambda t: (-t[0], -t[1]))
        return prefs + [f"Saved fact: {c}" for _, _, c in scored[:limit]]

    def clear(self) -> None:
        self.conn.execute("DELETE FROM facts")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()
