"""
Run self-play games between a fixed Dragapult heuristic agent and a list of
opponent agent dirs, recording every decision the Dragapult side faced
(observation dict before the decision, chosen action indices) plus the game
result. Output: one JSON file per game (only for games Dragapult WON) into
--out-dir, schema:
  {"winner": 0, "opponent_dir": "...", "decisions": [{"observation": {...}, "action": [...]}]}

Run from repo root with the project venv active:
  python3 agent/bc_dragapult_v3_combined/selfplay_record.py <dragapult_dir> <out_dir> <games_per_opponent> <opponent_dir1> [<opponent_dir2> ...]
"""
import json
import os
import sys

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "harness")
sys.path.insert(0, HARNESS)
from run_match import load_agent, load_deck  # noqa: E402
from cg.game import battle_start, battle_select, battle_finish  # noqa: E402


def run_match_recording(agent0, deck0, agent1, deck1, dragapult_index, max_selects=5000):
    """Like run_match.run_match but records dragapult_index's own decisions."""
    obs, start = battle_start(deck0, deck1)
    if start.errorPlayer >= 0:
        raise RuntimeError(f"Deck error: player {start.errorPlayer}, errorType {start.errorType}")
    decisions = []
    selects = 0
    while obs["current"]["result"] < 0:
        your_index = obs["current"]["yourIndex"]
        if your_index == dragapult_index:
            sel = obs.get("select")
            n_opts = len(sel["option"]) if sel and sel.get("option") else 0
            choice = agent0(obs) if your_index == 0 else agent1(obs)
            if n_opts >= 2:
                # store a shallow copy -- obs dict is rebuilt fresh each call by cg.game
                decisions.append({"observation": obs, "action": list(choice)})
        else:
            choice = agent0(obs) if your_index == 0 else agent1(obs)
        obs = battle_select(choice)
        selects += 1
        if selects > max_selects:
            battle_finish()
            raise RuntimeError("exceeded max_selects")
    result = obs["current"]["result"]
    battle_finish()
    return result, decisions


def main():
    dragapult_dir = os.path.abspath(sys.argv[1])
    out_dir = os.path.abspath(sys.argv[2])
    n_per_opponent = int(sys.argv[3])
    opponent_dirs = [os.path.abspath(p) for p in sys.argv[4:]]
    os.makedirs(out_dir, exist_ok=True)

    drag_mod = load_agent("drag_agent", os.path.join(dragapult_dir, "main.py"), chdir_for_import=dragapult_dir)
    drag_deck = load_deck(os.path.join(dragapult_dir, "deck.csv"))

    total_wins = 0
    total_games = 0
    for opp_dir in opponent_dirs:
        opp_name = os.path.basename(opp_dir)
        opp_mod = load_agent(f"opp_{opp_name}", os.path.join(opp_dir, "main.py"), chdir_for_import=opp_dir)
        opp_deck = load_deck(os.path.join(opp_dir, "deck.csv"))

        wins_this_opp = 0
        for i in range(n_per_opponent):
            # alternate which side is "player 0" to avoid any positional bias
            drag_index = i % 2
            total_games += 1
            try:
                if drag_index == 0:
                    result, decisions = run_match_recording(drag_mod.agent, drag_deck, opp_mod.agent, opp_deck, 0)
                else:
                    result, decisions = run_match_recording(opp_mod.agent, opp_deck, drag_mod.agent, drag_deck, 1)
            except Exception as e:
                print(f"  [{opp_name}] game {i}: ERROR {type(e).__name__}: {e}", flush=True)
                continue

            if result == drag_index and decisions:
                wins_this_opp += 1
                total_wins += 1
                out_path = os.path.join(out_dir, f"{opp_name}_{i}.json")
                with open(out_path, "w") as f:
                    json.dump({"winner": drag_index, "opponent": opp_name, "decisions": decisions}, f)

            if (i + 1) % 100 == 0:
                print(f"  [{opp_name}] {i+1}/{n_per_opponent} done, wins_so_far={wins_this_opp}", flush=True)

        print(f"{opp_name}: {wins_this_opp}/{n_per_opponent} Dragapult wins recorded", flush=True)

    print(f"TOTAL: {total_wins}/{total_games} Dragapult win-games recorded to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
