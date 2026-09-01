"""
Offline extraction of (feature_vector, label, group_id) rows from real Kaggle
episode replays, for training a behavior-cloning model.

For every episode file, for every decision point steps[i-1][p].observation.select
made by the eventual WINNER of that game (rewards[p] == max(rewards) and strictly
greater than the other player's reward), we build one row per candidate Option:
label=1 if that option's index is in steps[i][p]['action'] (the winner's actual
choice), else 0. Rows belonging to the same decision share a group_id so we can
compute decision-level (argmax) accuracy at eval time, and we split by EPISODE
(not by decision) into train/held-out to avoid leakage.

Output: a single JSON-lines file, one row per candidate option:
  {"episode": <file>, "group": <int>, "label": 0/1, "feat": [.. FEATURE_DIM floats ..],
   "n_options": <int>, "select_type": int, "select_context": int}

Run from agent/bc_v1/ with the project venv active.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cg.api import all_card_data, all_attack, to_observation_class
from features import extract_features, FEATURE_DIM

RESEARCH_EPISODES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "research", "episodes"
)


def find_episode_files():
    pattern = os.path.join(RESEARCH_EPISODES, "**", "*.json")
    files = sorted(glob.glob(pattern, recursive=True))
    return files


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "extracted_rows.jsonl"

    all_card = all_card_data()
    card_table = {c.cardId: c for c in all_card}
    all_atk = all_attack()
    attack_table = {a.attackId: a for a in all_atk}

    files = find_episode_files()
    print(f"Found {len(files)} episode files")

    n_games = 0
    n_games_used = 0
    n_decisions = 0
    n_rows = 0
    n_errors = 0
    group_counter = 0

    dragapult_id = 121

    with open(out_path, "w") as out_f:
        for fp in files:
            try:
                with open(fp) as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  SKIP {fp}: failed to load ({e})")
                continue
            n_games += 1

            rewards = data.get("rewards")
            steps = data.get("steps")
            if not rewards or not steps or len(rewards) != 2:
                continue
            if rewards[0] == rewards[1]:
                continue  # draw or no result, skip
            winner_idx = 0 if rewards[0] > rewards[1] else 1

            # Best-effort archetype tag for the winner (informational only).
            winner_has_dragapult = False
            try:
                step0_viz = steps[0][0]["visualize"][0]["action"]
                winner_deck = step0_viz[winner_idx]
                winner_has_dragapult = dragapult_id in winner_deck
            except Exception:
                pass

            used_this_game = False
            for i in range(1, len(steps)):
                for p in range(2):
                    if p != winner_idx:
                        continue
                    prev_obs_dict = steps[i - 1][p]["observation"]
                    sel = prev_obs_dict.get("select")
                    if sel is None:
                        continue
                    options = sel.get("option") or []
                    if len(options) < 2:
                        continue  # forced/trivial decision, no discriminative signal
                    chosen = steps[i][p]["action"]
                    if chosen is None:
                        continue
                    chosen_set = set(chosen)
                    if not chosen_set or not all(0 <= c < len(options) for c in chosen_set):
                        continue

                    try:
                        obs = to_observation_class(prev_obs_dict)
                    except Exception:
                        n_errors += 1
                        continue
                    if obs.select is None or obs.current is None:
                        continue

                    group_counter += 1
                    n_decisions += 1
                    used_this_game = True
                    rows_this_decision = []
                    ok = True
                    for oi, opt in enumerate(obs.select.option):
                        try:
                            feat = extract_features(obs, opt, card_table, attack_table)
                        except Exception:
                            ok = False
                            break
                        label = 1 if oi in chosen_set else 0
                        rows_this_decision.append({
                            "episode": os.path.basename(fp),
                            "game_key": fp,
                            "group": group_counter,
                            "label": label,
                            "feat": feat,
                            "n_options": len(options),
                            "select_type": int(obs.select.type) if obs.select.type is not None else -1,
                            "select_context": int(obs.select.context) if obs.select.context is not None else -1,
                            "winner_has_dragapult": winner_has_dragapult,
                        })
                    if not ok:
                        n_errors += 1
                        continue
                    for r in rows_this_decision:
                        out_f.write(json.dumps(r) + "\n")
                        n_rows += 1
            if used_this_game:
                n_games_used += 1

    print(f"Games seen: {n_games}, games with usable winner decisions: {n_games_used}")
    print(f"Decisions (groups) extracted: {n_decisions}, rows written: {n_rows}, decode/feature errors: {n_errors}")
    print(f"FEATURE_DIM = {FEATURE_DIM}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
