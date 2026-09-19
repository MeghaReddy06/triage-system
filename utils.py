"""
utils.py
Utility functions for loading input CSV and saving output CSV.

v3: No logic changes from v2. Strict schema validation, debug JSON export,
LoadResult dataclass, and empty-row stats all already in place.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Column candidates
# ---------------------------------------------------------------------------

TEXT_FIELD_CANDIDATES = [
    "Issue", "Subject", "Description", "message", "text", "query", "body", "content",
]

ID_FIELD_CANDIDATES = [
    "ticket_id", "id", "TicketID",
]

REQUIRED_OUTPUT_FIELDS = {
    "ticket_id", "request_type", "status", "product_area", "response", "justification",
}

VALID_STATUSES = {"replied", "escalated"}

OUTPUT_FIELDS_ORDER = [
    "ticket_id", "request_type", "status", "product_area", "response", "justification",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class LoadResult:
    tickets:        list[dict] = field(default_factory=list)
    loaded:         int = 0
    skipped_empty:  int = 0
    skipped_no_id:  int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_column(fieldnames: list[str], candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in fieldnames:
            return c
    return None


def _validate_row(row: dict, row_index: int) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_OUTPUT_FIELDS - set(row.keys())
    if missing:
        errors.append(f"row {row_index}: missing fields {sorted(missing)}")
    status = row.get("status", "")
    if status not in VALID_STATUSES:
        errors.append(f"row {row_index}: invalid status '{status}'")
    for f in ("ticket_id", "request_type", "product_area"):
        if not str(row.get(f, "")).strip():
            errors.append(f"row {row_index}: '{f}' is empty")
    return errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_tickets(file_path: str) -> LoadResult:
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    result = LoadResult()

    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            print("[ERROR] CSV has no headers.")
            sys.exit(1)

        fieldnames = list(reader.fieldnames)
        id_col = find_column(fieldnames, ID_FIELD_CANDIDATES)
        if id_col is None:
            print("[WARN] No ticket_id column found. Generating IDs automatically.")

        available_text_fields = [col for col in TEXT_FIELD_CANDIDATES if col in fieldnames]
        if not available_text_fields:
            print("[ERROR] No usable text fields found in dataset.")
            sys.exit(1)

        for row_num, row in enumerate(reader, start=1):
            if id_col and row.get(id_col, "").strip():
                ticket_id = row[id_col].strip()
            else:
                ticket_id = f"ROW_{row_num}"
                result.skipped_no_id += 1

            text_parts: list[str] = []
            for f in ["Subject", "Issue"]:
                val = row.get(f, "").strip()
                if val:
                    text_parts.append(val)

            if not text_parts:
                for f in available_text_fields:
                    val = row.get(f, "").strip()
                    if val:
                        text_parts.append(val)

            text = " ".join(text_parts).strip()

            if not text:
                print(f"[WARN] Row {row_num} ({ticket_id}) has no usable text — skipping.")
                result.skipped_empty += 1
                continue

            result.tickets.append({"ticket_id": ticket_id, "text": text})
            result.loaded += 1

    return result


def save_results(
    file_path: str,
    results: list[dict],
    debug: bool = False,
) -> None:
    """
    Validate and write results to CSV.
    If debug=True, also writes <basename>.debug.json with full _trace records.
    Internal _trace key is stripped from CSV output.
    """
    if not results:
        print("[WARN] No results to save.")
        return

    out_dir = os.path.dirname(file_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    all_errors: list[str] = []
    for i, row in enumerate(results):
        all_errors.extend(_validate_row(row, i + 1))

    if all_errors:
        print("[ERROR] Output validation failed:")
        for err in all_errors:
            print(f"         {err}")
        print("[ERROR] Aborting save to prevent corrupted output.")
        sys.exit(1)

    with open(file_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS_ORDER, extrasaction="ignore")
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in OUTPUT_FIELDS_ORDER})

    print(f"[INFO] Results saved -> {file_path}")

    if debug:
        debug_path = os.path.splitext(file_path)[0] + ".debug.json"
        debug_records = []
        for row in results:
            record = {k: v for k, v in row.items() if k != "_trace"}
            record["_trace"] = row.get("_trace", {})
            debug_records.append(record)
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(debug_records, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Debug trace  -> {debug_path}")