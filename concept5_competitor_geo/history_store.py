"""Persist weekly visibility snapshots in SQLite for trend tracking."""
from __future__ import annotations
import sqlite3
from datetime import date
from pathlib import Path
from visibility_scorer import BrandVisibility, VisibilityReport
from logger import get_logger

log = get_logger("history_store")
DB_PATH = Path("data/geo_history.db")


class HistoryStore:
    def __init__(self):
        DB_PATH.parent.mkdir(exist_ok=True)
        self._conn = sqlite3.connect(str(DB_PATH))
        self._init_schema()

    def _init_schema(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS visibility_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_date TEXT NOT NULL,
                category TEXT NOT NULL,
                brand TEXT NOT NULL,
                mention_count INTEGER,
                total_probes INTEGER,
                visibility_pct REAL,
                models_mentioned TEXT
            )
        """)
        self._conn.commit()

    def save(self, report: VisibilityReport) -> None:
        today = date.today().isoformat()
        rows = [
            (today, report.category, b.brand, b.mention_count,
             b.total_probes, b.visibility_pct, ",".join(b.models_mentioned))
            for b in report.brands
        ]
        self._conn.executemany(
            "INSERT INTO visibility_snapshots "
            "(snapshot_date,category,brand,mention_count,total_probes,visibility_pct,models_mentioned) "
            "VALUES (?,?,?,?,?,?,?)",
            rows,
        )
        self._conn.commit()
        log.info("Saved %d brand snapshots for '%s' on %s", len(rows), report.category, today)

    def get_trend(self, brand: str, category: str, weeks: int = 8) -> list[dict]:
        cur = self._conn.execute(
            "SELECT snapshot_date, visibility_pct FROM visibility_snapshots "
            "WHERE brand=? AND category=? ORDER BY snapshot_date DESC LIMIT ?",
            (brand, category, weeks),
        )
        return [{"date": row[0], "visibility_pct": row[1]} for row in cur.fetchall()]

    def close(self):
        self._conn.close()
