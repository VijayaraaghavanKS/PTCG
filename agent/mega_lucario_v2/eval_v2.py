"""Evaluation harness for mega_lucario_v2: A/B vs ga_v1 (20 games), plus
dragapult_day1 (16 games) and iono_day1 (16 games). Alternates who goes first.
Also aggregates SEARCH_STATS and per-decision wall-clock timing."""
import os
import sys
import time
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HARNESS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "harness"))
sys.path.insert(0, HARNESS_DIR)
from run_match import load_agent, load_deck, run_match  # noqa: E402

MY_DIR = BASE_DIR
OPPONENTS = [
    (os.path.abspath(os.path.join(BASE_DIR, "..", "ga_v1")), 60),
    (os.path.abspath(os.path.join(BASE_DIR, "..", "dragapult_day1")), 60),
    (os.path.abspath(os.path.join(BASE_DIR, "..", "iono_day1")), 100),
]


def run_eval(opp_dir, n_games, tag):
    my_mod = load_agent(f"v2_{tag}", os.path.join(MY_DIR, "main.py"), chdir_for_import=MY_DIR)
    my_deck = load_deck(os.path.join(MY_DIR, "deck.csv"))
    opp_name = os.path.basename(opp_dir)
    opp_mod = load_agent(f"opp_{tag}", os.path.join(opp_dir, "main.py"), chdir_for_import=opp_dir)
    opp_deck = load_deck(os.path.join(opp_dir, "deck.csv"))

    wins = 0
    total_selects = 0
    t_start = time.time()
    per_game = []
    for g in range(n_games):
        v2_first = (g % 2 == 0)
        try:
            g_t0 = time.time()
            if v2_first:
                result, selects = run_match(my_mod.agent, my_deck, opp_mod.agent, opp_deck)
                win = (result == 0)
            else:
                result, selects = run_match(opp_mod.agent, opp_deck, my_mod.agent, my_deck)
                win = (result == 1)
            g_t1 = time.time()
            wins += int(win)
            total_selects += selects
            per_game.append({"game": g, "win": win, "selects": selects, "time_s": g_t1 - g_t0})
            print(f"  [{opp_name}] game {g}: {'WIN' if win else 'loss'} selects={selects} time={g_t1-g_t0:.2f}s "
                  f"v2_first={v2_first}")
        except Exception as e:
            print(f"  [{opp_name}] game {g}: ERROR {type(e).__name__}: {e}")
            per_game.append({"game": g, "error": str(e)})
    t_end = time.time()
    stats = dict(my_mod.SEARCH_STATS)
    print(f"  [{opp_name}] TOTAL: {wins}/{n_games} ({100.0*wins/n_games:.1f}%)  "
          f"wall={t_end-t_start:.1f}s  search_stats={stats}")
    return {
        "opponent": opp_name,
        "wins": wins,
        "games": n_games,
        "win_rate": wins / n_games,
        "wall_time_s": t_end - t_start,
        "total_selects": total_selects,
        "search_stats": stats,
        "per_game": per_game,
    }


def main():
    results = []
    for opp_dir, n in OPPONENTS:
        tag = os.path.basename(opp_dir)
        print(f"=== vs {tag} ({n} games) ===")
        results.append(run_eval(opp_dir, n, tag))
    total_wins = sum(r["wins"] for r in results)
    total_games = sum(r["games"] for r in results)
    print(f"\nBLENDED: {total_wins}/{total_games} ({100.0*total_wins/total_games:.1f}%)")
    with open(os.path.join(BASE_DIR, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("Written to eval_results.json")


if __name__ == "__main__":
    main()
