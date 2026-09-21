import sqlite3
from contextlib import contextmanager

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    post_id TEXT NOT NULL UNIQUE,
    permalink TEXT,
    text TEXT,
    score INTEGER,
    matched_keywords TEXT,
    matched_categories TEXT,
    suspicious_urls TEXT,
    is_likely_phishing INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


@contextmanager
def get_connection():
    conn = sqlite3.connect(config.DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def init_db() -> None:
    import os

    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    with get_connection() as conn:
        conn.execute(SCHEMA)
        conn.commit()


def save_result(source: str, post_id: str, permalink: str, analysis) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO results
                (source, post_id, permalink, text, score, matched_keywords,
                 matched_categories, suspicious_urls, is_likely_phishing)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source,
                post_id,
                permalink,
                analysis.text,
                analysis.score,
                ",".join(analysis.matched_keywords),
                ",".join(analysis.matched_categories),
                ",".join(analysis.suspicious_urls),
                int(analysis.is_likely_phishing),
            ),
        )
        conn.commit()
