"""Local match runner: pits two main.py-style agents against each other via the cg engine."""
import importlib.util
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)  # so `import cg` resolves to harness/cg

from cg.game import battle_start, battle_select, battle_finish  # noqa: E402


def load_agent(module_name: str, file_path: str, chdir_for_import: str | None = None):
    """Import a main.py as a module. chdir_for_import lets module-level deck.csv reads resolve."""
    cwd = os.getcwd()
    if chdir_for_import:
        os.chdir(chdir_for_import)
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
    return mod


def load_deck(path: str) -> list[int]:
    with open(path) as f:
        lines = [line.strip() for line in f if line.strip()]
    deck = [int(x) for x in lines[:60]]
    if len(deck) != 60:
        raise ValueError(f"{path}: expected 60 cards, got {len(deck)}")
    return deck


def run_match(agent0, deck0, agent1, deck1, max_selects=5000):
    obs, start = battle_start(deck0, deck1)
    if start.errorPlayer >= 0:
        raise RuntimeError(f"Deck error: player {start.errorPlayer}, errorType {start.errorType}")
    selects = 0
    while obs["current"]["result"] < 0:
        your_index = obs["current"]["yourIndex"]
        choice = agent0(obs) if your_index == 0 else agent1(obs)
        obs = battle_select(choice)
        selects += 1
        if selects > max_selects:
            battle_finish()
            raise RuntimeError("exceeded max_selects, possible infinite loop")
    result = obs["current"]["result"]
    battle_finish()
    return result, selects


if __name__ == "__main__":
    dragapult_dir = os.path.join(BASE, "..", "dragapult_day1")
    dragapult_mod = load_agent("dragapult_agent", os.path.join(dragapult_dir, "main.py"), chdir_for_import=dragapult_dir)
    dragapult_deck = load_deck(os.path.join(dragapult_dir, "deck.csv"))

    random_mod = load_agent("random_agent_mod", os.path.join(BASE, "random_agent.py"))
    random_deck = load_deck(os.path.join(BASE, "random_deck.csv"))

    N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    tally = {0: 0, 1: 0, 2: 0}
    for i in range(N):
        try:
            result, selects = run_match(dragapult_mod.agent, dragapult_deck, random_mod.agent, random_deck)
            tally[result] = tally.get(result, 0) + 1
            print(f"game {i}: winner_index={result} (0=dragapult,1=random,2=draw) selects={selects}")
        except Exception as e:
            print(f"game {i}: ERROR {type(e).__name__}: {e}")

    print(f"\nDragapult wins={tally.get(0,0)} Random wins={tally.get(1,0)} Draws={tally.get(2,0)} / {N}")
