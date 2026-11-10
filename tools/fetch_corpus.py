#!/usr/bin/env python3
"""Fetch finished games from a govariants server as a rules-conformance corpus.

The Rust engine is a second implementation of rules that already exist in
TypeScript, so the two will drift. This downloads real played games -- the
board after every single round, as the server computed it -- so the Rust port
can be replayed against them offline.

`state?round=N` reports `last_move` for that round, so the move sequence is
recoverable for every board type, not just the rectangular ones that SGF
export supports.

Usage:
    python3 tools/fetch_corpus.py --board-type grid --games 6
    python3 tools/fetch_corpus.py --board-type circular --games 3 --min-rounds 20
"""

import argparse
import json
import pathlib
import subprocess
import sys
import time

DEFAULT_BASE = "https://www.govariants.com/api"
# Boards are stored as glyph strings rather than nested integer arrays: a
# 199-round 19x19 game is 637 KB as JSON integers and under 90 KB this way,
# which is the difference between a committable corpus and an awkward one.
GLYPHS = ".XO#*+"
# Be a polite client: this is someone's live server, not a test fixture.
DELAY_SECONDS = 0.25


def compact_board(board):
    """2D board -> one string per row; 1D (graph) board -> a single string.

    Falls back to the raw array if a variant uses colors outside GLYPHS.
    """
    if board is None:
        return None
    if board and isinstance(board[0], list):
        if all(isinstance(c, int) and 0 <= c < len(GLYPHS) for row in board for c in row):
            return ["".join(GLYPHS[c] for c in row) for row in board]
        return board
    if all(isinstance(c, int) and 0 <= c < len(GLYPHS) for c in board):
        return "".join(GLYPHS[c] for c in board)
    return board


def fetch(base, path, raw=False, timeout=25):
    result = subprocess.run(
        ["curl", "-s", "--max-time", str(timeout), f"{base}{path}"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed for {path}: {result.stderr.strip()}")
    if raw:
        return result.stdout
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"bad JSON from {path}: {exc}") from exc


def list_candidates(base, variant, board_type, min_rounds, pages, per_page):
    seen = []
    for offset in range(0, pages * per_page, per_page):
        games = fetch(
            base,
            f"/games?variant={variant}&count={per_page}&offset={offset}",
        )
        if not isinstance(games, list) or not games:
            break
        for game in games:
            board = (game.get("config") or {}).get("board") or {}
            if board.get("type") != board_type:
                continue
            if game.get("phase") != "gameover":
                continue
            if (game.get("round") or 0) < min_rounds:
                continue
            seen.append(game)
        time.sleep(DELAY_SECONDS)
    return seen


def fetch_game(base, game_id):
    """Download a game plus the board after every round."""
    initial = fetch(base, f"/games/{game_id}/state/initial")
    total_rounds = initial.get("round") or 0

    rounds = []
    for round_index in range(total_rounds + 1):
        state = fetch(base, f"/games/{game_id}/state?round={round_index}&seat=")
        inner = state.get("state") or {}
        rounds.append(
            {
                "round": round_index,
                "last_move": inner.get("last_move", ""),
                "board": compact_board(inner.get("board")),
                "captures": inner.get("captures"),
                "next_to_play": state.get("next_to_play"),
            }
        )
        time.sleep(DELAY_SECONDS)

    return {
        "id": game_id,
        "variant": initial.get("variant"),
        "config": initial.get("config"),
        "result": initial.get("result"),
        "phase": initial.get("phase"),
        "total_rounds": total_rounds,
        "special_moves": initial.get("special_moves"),
        "glyphs": GLYPHS,
        "rounds": rounds,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE, help="API base URL")
    parser.add_argument("--variant", default="baduk")
    parser.add_argument("--board-type", default="grid")
    parser.add_argument("--games", type=int, default=6, help="how many to keep")
    parser.add_argument("--min-rounds", type=int, default=15)
    parser.add_argument("--pages", type=int, default=4)
    parser.add_argument("--per-page", type=int, default=100)
    parser.add_argument(
        "--out",
        default="tests/corpus",
        help="directory to write <variant>-<board>-<id>.json files into",
    )
    args = parser.parse_args()

    print(
        f"searching {args.base} for finished {args.variant} games "
        f"on {args.board_type} boards with >= {args.min_rounds} rounds..."
    )
    candidates = list_candidates(
        args.base,
        args.variant,
        args.board_type,
        args.min_rounds,
        args.pages,
        args.per_page,
    )
    print(f"  {len(candidates)} candidates")
    if not candidates:
        print("nothing to fetch", file=sys.stderr)
        return 1

    # Prefer longer games: more rounds means more chances to catch a divergence.
    candidates.sort(key=lambda g: -(g.get("round") or 0))
    chosen = candidates[: args.games]

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for game in chosen:
        game_id = game["id"]
        rounds = game.get("round")
        print(f"  fetching {game_id} ({rounds} rounds)...", end="", flush=True)
        try:
            record = fetch_game(args.base, game_id)
        except RuntimeError as exc:
            print(f" FAILED: {exc}")
            continue
        name = f"{record['variant']}-{args.board_type}-{game_id}.json"
        path = out_dir / name
        path.write_text(json.dumps(record, indent=1))
        print(f" wrote {path} ({path.stat().st_size // 1024} KB)")

    print(f"\ncorpus now: {len(list(out_dir.glob('*.json')))} games in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
