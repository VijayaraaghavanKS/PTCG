"""Driver: run N games for each (agentA, agentB) pair and print a clean summary
including a crash count (exceptions caught per game)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_match import load_agent, load_deck, run_match  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
AGENT_ROOT = os.path.join(HERE, "..")


def run_pair(dirA, dirB, n):
    dirA = os.path.join(AGENT_ROOT, dirA)
    dirB = os.path.join(AGENT_ROOT, dirB)
    modA = load_agent(f"agentA_{os.path.basename(dirA)}", os.path.join(dirA, "main.py"), chdir_for_import=dirA)
    deckA = load_deck(os.path.join(dirA, "deck.csv"))
    modB = load_agent(f"agentB_{os.path.basename(dirB)}", os.path.join(dirB, "main.py"), chdir_for_import=dirB)
    deckB = load_deck(os.path.join(dirB, "deck.csv"))

    tally = {0: 0, 1: 0, 2: 0}
    crashes = 0
    t0 = time.time()
    for i in range(n):
        try:
            result, selects = run_match(modA.agent, deckA, modB.agent, deckB)
            tally[result] = tally.get(result, 0) + 1
        except Exception as e:
            crashes += 1
            print(f"    CRASH game {i}: {type(e).__name__}: {e}")
    dt = time.time() - t0
    a_wins, b_wins, draws = tally.get(0, 0), tally.get(1, 0), tally.get(2, 0)
    pct = 100.0 * a_wins / n if n else 0.0
    print(f"  {os.path.basename(dirA)} vs {os.path.basename(dirB)}: "
          f"{a_wins}/{n} ({pct:.1f}%) A-wins, {b_wins} B-wins, {draws} draws, "
          f"{crashes} crashes, {dt:.1f}s total ({1000*dt/n:.0f}ms/game)")
    return a_wins, b_wins, draws, crashes


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 150
    pairs = [
        ("bc_v2", "dragapult_day1"),
        ("bc_v2", "iono_day1"),
        ("bc_v2", "mega_lucario_ref"),
        ("bc_reranker_v2", "dragapult_day1"),
        ("bc_reranker_v2", "iono_day1"),
        ("bc_reranker_v2", "mega_lucario_ref"),
    ]
    total_crashes = 0
    total_games = 0
    for a, b in pairs:
        _, _, _, c = run_pair(a, b, n)
        total_crashes += c
        total_games += n
    print(f"\nTOTAL: {total_games} games, {total_crashes} crashes")
