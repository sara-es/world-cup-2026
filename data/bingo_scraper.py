#!/usr/bin/env python3
"""
Scrape worldcupbingo hundreds.html and write a CSV of board_id, team_1..team_N.

Dependencies: requests, beautifulsoup4
    pip install requests beautifulsoup4
"""

import csv
import sys
import uuid

import requests
from bs4 import BeautifulSoup

URL = "https://stain.github.io/worldcupbingo/hundreds.html"
OUTPUT = "data/bingo_cards.csv"

NAMESPACE_WORLDCUP = uuid.UUID("dedaeff9-2834-51b1-afda-9d8e2ea53d38")


def compute_board_id(teams: list[str]) -> str:
    """Replicates getBoardHash() from worldcupbingo.py."""
    return str(uuid.uuid5(NAMESPACE_WORLDCUP, "\n".join(sorted(teams))))


def main():
    print(f"Fetching {URL} ...", file=sys.stderr)
    r = requests.get(URL, timeout=30)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, "html.parser")
    boards = soup.find_all("div", class_="bingo")
    print(f"Found {len(boards)} boards.", file=sys.stderr)

    if not boards:
        sys.exit("No boards found – the page structure may have changed.")

    records = []
    mismatches = 0

    for i, board in enumerate(boards, start=1):
        # Board ID embedded in the page (two places; use the h1 id)
        h1 = board.find("h1")
        embedded_id = h1["id"].strip() if (h1 and h1.get("id")) else None

        # Fallback: read from the <div class='hash'><b>...</b></div>
        if not embedded_id:
            hash_div = board.find("div", class_="hash")
            b_tag = hash_div.find("b") if hash_div else None
            embedded_id = b_tag.get_text(strip=True) if b_tag else None

        # Teams: text content of each <td> in the board's table
        table = board.find("table")
        if not table:
            print(f"  Board {i}: no table found, skipping.", file=sys.stderr)
            continue
        teams = [td.get_text(strip=True) for td in table.find_all("td")]

        # Cross-check by recomputing the ID from the teams
        computed_id = compute_board_id(teams)
        if embedded_id and computed_id != embedded_id:
            print(
                f"  WARNING board {i}: embedded ID {embedded_id!r} != "
                f"computed {computed_id!r}",
                file=sys.stderr,
            )
            mismatches += 1

        board_id = embedded_id or computed_id
        records.append((board_id, teams))

    if mismatches:
        print(
            f"\n{mismatches} ID mismatches – check the WARNING lines above.",
            file=sys.stderr,
        )

    # Determine max number of teams across all boards (should be uniform)
    max_teams = max(len(r[1]) for r in records)
    team_headers = [f"team_{i}" for i in range(1, max_teams + 1)]

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["board_id"] + team_headers)
        for board_id, teams in records:
            # Pad with empty strings if any board has fewer teams than max_teams
            row = [board_id] + teams + [""] * (max_teams - len(teams))
            writer.writerow(row)

    print(f"\nWrote {len(records)} rows to {OUTPUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
