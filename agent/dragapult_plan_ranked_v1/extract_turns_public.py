"""Same turn-level extraction as extract_turns.py, but scanning the public
top-episode-index dataset for Dragapult-ex-archetype WINS (winner's side
shows Dreepy/Drakloak/Dragapult ex, card ids 119/120/121, anywhere in
active/bench/discard across the game)."""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_turns import extract_turns_for_side, load_game

DRAGAPULT_LINE_IDS = {119, 120, 121}


def side_has_dragapult(d, idx):
    steps = d["steps"]
    for step in steps:
        try:
            cur = step[idx]["observation"].get("current")
        except Exception:
            continue
        if not isinstance(cur, dict):
            continue
        try:
            side = cur["players"][idx]
        except Exception:
            continue
        for area_name in ("active", "bench", "discard"):
            for p in side.get(area_name) or []:
                if isinstance(p, dict) and p.get("id") in DRAGAPULT_LINE_IDS:
                    return True
                if isinstance(p, int) and p in DRAGAPULT_LINE_IDS:
                    return True
    return False


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "turn_rows_public.jsonl"
    n_games = 0
    n_rows = 0
    n_scanned = 0
    with open(out_path, "w") as out:
        for fp in sorted(glob.glob("research/episodes/*/*.json")) + sorted(glob.glob("research/episodes/*.json")):
            n_scanned += 1
            try:
                d = load_game(fp)
            except Exception:
                continue
            rewards = d.get("rewards", [0, 0])
            if not isinstance(rewards, list) or len(rewards) != 2:
                continue
            winner = 0 if rewards[0] == 1 else (1 if rewards[1] == 1 else None)
            if winner is None:
                continue
            try:
                if not side_has_dragapult(d, winner):
                    continue
            except Exception:
                continue
            try:
                rows = extract_turns_for_side(d, winner)
            except Exception:
                continue
            if not rows:
                continue
            n_games += 1
            for r in rows:
                r["source"] = "public"
                r["game"] = os.path.basename(fp)
                out.write(json.dumps(r) + "\n")
                n_rows += 1
    print(f"public episodes scanned: {n_scanned}, dragapult win-games: {n_games}, turn rows: {n_rows}")


if __name__ == "__main__":
    main()
