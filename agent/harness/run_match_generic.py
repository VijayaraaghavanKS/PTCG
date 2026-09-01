"""Run N games between any two agent dirs (each with main.py, deck.csv)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_match import load_agent, load_deck, run_match  # noqa: E402

def main():
    dirA, dirB = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    modA = load_agent("agentA", os.path.join(dirA, "main.py"), chdir_for_import=dirA)
    deckA = load_deck(os.path.join(dirA, "deck.csv"))
    modB = load_agent("agentB", os.path.join(dirB, "main.py"), chdir_for_import=dirB)
    deckB = load_deck(os.path.join(dirB, "deck.csv"))

    tally = {0: 0, 1: 0, 2: 0}
    for i in range(N):
        try:
            result, selects = run_match(modA.agent, deckA, modB.agent, deckB)
            tally[result] = tally.get(result, 0) + 1
            print(f"game {i}: winner_index={result} (0={dirA},1={dirB}) selects={selects}")
        except Exception as e:
            print(f"game {i}: ERROR {type(e).__name__}: {e}")
    print(f"\n{dirA} wins={tally.get(0,0)} {dirB} wins={tally.get(1,0)} draws={tally.get(2,0)} / {N}")

if __name__ == "__main__":
    main()
