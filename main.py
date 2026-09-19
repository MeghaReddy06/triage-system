"""
main.py
Entry point for the Support Ticket Triage Agent.

v3: No logic changes from v2.
--debug flag, LoadResult stats, per-ticket kb_score display, and
per-category summary already in place.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from agent import TriageAgent
from retriever import KnowledgeBase
from utils import load_tickets, save_results

DEFAULT_INPUT     = os.path.join(_PROJECT_ROOT, "support_tickets", "support_tickets.csv")
DEFAULT_OUTPUT    = os.path.join(_PROJECT_ROOT, "support_tickets", "output.csv")
DEFAULT_DATA_BASE = _PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Support Ticket Triage Agent")
    parser.add_argument("--input",  default=DEFAULT_INPUT,     help="Path to input CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,    help="Path to output CSV")
    parser.add_argument("--data",   default=DEFAULT_DATA_BASE, help="Root path for KB data dirs")
    parser.add_argument(
        "--debug", action="store_true",
        help="Export per-ticket reasoning trace to <output>.debug.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 65)
    print("  Support Ticket Triage Agent")
    print("=" * 65)
    print(f"  Input  : {args.input}")
    print(f"  Output : {args.output}")
    print(f"  Data   : {args.data}")
    print(f"  Debug  : {'on' if args.debug else 'off'}")
    print("=" * 65)

    # 1. Load
    print("\n[1/4] Loading tickets ...")
    load_result = load_tickets(args.input)

    if load_result.loaded == 0:
        print("[ERROR] No valid tickets found. Exiting.")
        sys.exit(1)

    print(f"      Loaded  : {load_result.loaded} ticket(s)")
    if load_result.skipped_empty:
        print(f"      Skipped : {load_result.skipped_empty} row(s) with no text")
    if load_result.skipped_no_id:
        print(f"      Auto-ID : {load_result.skipped_no_id} row(s) without ticket_id")

    # 2. Build KB
    print("\n[2/4] Building knowledge base ...")
    t0 = time.time()
    kb = KnowledgeBase(base_path=args.data)
    print(f"      Indexed {len(kb.passages)} passage(s) in {time.time() - t0:.2f}s.")
    if len(kb.passages) == 0:
        print("[WARNING] No KB files found — running in keyword-only fallback mode.")

    # 3. Process
    print(f"\n[3/4] Processing {load_result.loaded} ticket(s) ...")
    print(f"  {'#':>5}  {'Ticket ID':<12} {'Product Area':<18} {'Status':<12} {'KB':>6}  {'Conf'}")
    print("  " + "-" * 62)

    agent   = TriageAgent(kb)
    results: list[dict] = []

    for i, ticket in enumerate(load_result.tickets, start=1):
        result = agent.process(ticket["ticket_id"], ticket["text"])
        results.append(result)

        trace    = result.get("_trace", {})
        kb_score = trace.get("kb_top_score", 0.0)
        cat_conf = trace.get("category_confidence", 0.0)
        tag      = " ESCALATE " if result["status"] == "escalated" else " REPLY  "

        print(
            f"  [{i:>4}]  {ticket['ticket_id']:<12} "
            f"{result['product_area']:<18} {tag:<12} "
            f"{kb_score:>6.3f}  {cat_conf:.3f}"
        )

    # 4. Save
    print("\n[4/4] Saving results ...")
    save_results(args.output, results, debug=args.debug)

    # Summary
    n_reply    = sum(1 for r in results if r["status"] == "replied")
    n_escalate = len(results) - n_reply
    area_counts = Counter(r["product_area"] for r in results)

    print("\n" + "=" * 65)
    print(f"  Total: {len(results)}   Replied: {n_reply}   Escalated: {n_escalate}")
    print("  By product area:")
    for area, count in sorted(area_counts.items(), key=lambda x: -x[1]):
        print(f"    {area:<22} {count}")
    print("=" * 65)


if __name__ == "__main__":
    main()