from __future__ import annotations
import csv
import os
from pathlib import Path
from datetime import datetime
from opportunity_finder import LinkOpportunity
from logger import get_logger

log = get_logger("exporter")

HEADERS = ["Priority", "Source Title", "Source URL", "Target Title", "Target URL",
           "Similarity", "Orphan Target", "Suggested Anchor"]

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False


class SheetsExporter:
    def export(self, opportunities: list[tuple[LinkOpportunity, str]]) -> str:
        if GSPREAD_AVAILABLE and Path("service_account.json").exists():
            return self._export_sheets(opportunities)
        return self._export_csv(opportunities)

    def _export_sheets(self, opportunities: list[tuple[LinkOpportunity, str]]) -> str:
        creds = Credentials.from_service_account_file(
            "service_account.json",
            scopes=["https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive"],
        )
        gc = gspread.authorize(creds)
        title = f"Internal Linking Opportunities {datetime.now().strftime('%Y-%m-%d')}"
        sheet = gc.create(title)
        ws = sheet.get_worksheet(0)
        rows = [HEADERS] + [
            [o.priority, o.source.title, o.source.url, o.target.title, o.target.url,
             f"{o.similarity:.3f}", str(o.is_orphan_target), anchor]
            for o, anchor in opportunities
        ]
        ws.update("A1", rows)
        ws.format("A1:H1", {"textFormat": {"bold": True}})
        ws.freeze(rows=1)
        url = f"https://docs.google.com/spreadsheets/d/{sheet.id}"
        log.info("Exported %d rows to Google Sheets: %s", len(opportunities), url)
        return url

    def _export_csv(self, opportunities: list[tuple[LinkOpportunity, str]]) -> str:
        Path("reports").mkdir(exist_ok=True)
        path = f"reports/linking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(HEADERS)
            for o, anchor in opportunities:
                writer.writerow([o.priority, o.source.title, o.source.url,
                                  o.target.title, o.target.url,
                                  f"{o.similarity:.3f}", o.is_orphan_target, anchor])
        log.info("Exported %d rows to %s", len(opportunities), path)
        return path
