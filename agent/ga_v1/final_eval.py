"""Final 48-game (16 each vs dragapult_day1/iono_day1/mega_lucario_ref) evaluation,
run once for the GA-evolved best weights and once for the hand-guessed DEFAULT_WEIGHTS
baseline (same architecture, no GA). Alternates who goes first each game."""
import os
import sys
import json
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HARNESS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "harness"))
sys.path.insert(0, HARNESS_DIR)
from run_match import load_agent, load_deck, run_match  # noqa: E402

sys.path.insert(0, BASE_DIR)
from main import DEFAULT_WEIGHTS  # noqa: E402
import genetic_search as gs  # reuse load_candidate_agent  # noqa: E402

OPPONENT_DIRS = [
    os.path.abspath(os.path.join(BASE_DIR, "..", "dragapult_day1")),
    os.path.abspath(os.path.join(BASE_DIR, "..", "iono_day1")),
    os.path.abspath(os.path.join(BASE_DIR, "..", "mega_lucario_ref")),
]
GAMES_PER_OPPONENT = 16


def run_eval(weights: dict, tag: str):
    mod, wpath = gs.load_candidate_agent(weights, tag)
    my_deck = load_deck(os.path.join(BASE_DIR, "deck.csv"))
    results = {}
    total_wins = 0
    total_games = 0
    for d in OPPONENT_DIRS:
        name = os.path.basename(d)
        opp_mod = load_agent(f"final_opp_{tag}_{name}", os.path.join(d, "main.py"), chdir_for_import=d)
        opp_deck = load_deck(os.path.join(d, "deck.csv"))
        wins = 0
        for g in range(GAMES_PER_OPPONENT):
            candidate_first = (g % 2 == 0)
            try:
                if candidate_first:
                    result, _ = run_match(mod.agent, my_deck, opp_mod.agent, opp_deck)
                    win = (result == 0)
                else:
                    result, _ = run_match(opp_mod.agent, opp_deck, mod.agent, my_deck)
                    win = (result == 1)
            except Exception as e:
                print(f"  ERROR game {g} vs {name}: {type(e).__name__}: {e}")
                win = False
            wins += int(win)
        results[name] = (wins, GAMES_PER_OPPONENT)
        total_wins += wins
        total_games += GAMES_PER_OPPONENT
        print(f"  vs {name}: {wins}/{GAMES_PER_OPPONENT}")
    try:
        os.remove(wpath)
    except OSError:
        pass
    results["TOTAL"] = (total_wins, total_games)
    print(f"  TOTAL: {total_wins}/{total_games} ({100.0*total_wins/total_games:.1f}%)")
    return results


def main():
    with open(os.path.join(BASE_DIR, "ga_best_weights.json")) as f:
        ga_weights = json.load(f)

    all_results = {}
    t0 = time.time()
    print(f"=== Final eval: GA-tuned weights ({GAMES_PER_OPPONENT} games/opponent) ===")
    all_results["ga_tuned"] = run_eval(ga_weights, "ga_final")
    t1 = time.time()
    print(f"(took {t1-t0:.1f}s)")

    print(f"\n=== Final eval: hand-guessed DEFAULT_WEIGHTS baseline ({GAMES_PER_OPPONENT} games/opponent) ===")
    all_results["baseline"] = run_eval(DEFAULT_WEIGHTS, "baseline_final")
    t2 = time.time()
    print(f"(took {t2-t1:.1f}s)")

    with open(os.path.join(BASE_DIR, "final_eval_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nWritten to final_eval_results.json")


if __name__ == "__main__":
    main()
