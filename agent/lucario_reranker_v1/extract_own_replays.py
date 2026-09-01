"""
Extract (feature, label, group) rows from OUR OWN real ladder replay wins
for the Mega Lucario ex line (pulled by pull_own_replays.py into
research/own_replays_lucario/). Same schema/logic as
bc_dragapult_v3_combined/extract_own_replays.py, retargeted.

Run from repo root with the project venv active:
  python3 agent/lucario_reranker_v1/extract_own_replays.py <own_replays_dir> <out_jsonl>
"""
import glob
import json
import os
import sys

BC_V2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bc_v2")
sys.path.insert(0, BC_V2)

from cg.api import all_card_data, all_attack, to_observation_class  # noqa: E402
from features import extract_features, FEATURE_DIM  # noqa: E402

OUR_TEAM_NAME = "K S VIJAYARAAGHAVAN"


def main():
    replay_dir = sys.argv[1]
    out_path = sys.argv[2]

    all_card = all_card_data()
    card_table = {c.cardId: c for c in all_card}
    all_atk = all_attack()
    attack_table = {a.attackId: a for a in all_atk}

    files = sorted(glob.glob(os.path.join(replay_dir, "*.json")))
    print(f"Found {len(files)} own-replay files")

    n_games = 0
    n_games_used = 0
    n_decisions = 0
    n_rows = 0
    n_errors = 0
    group_counter = 0

    with open(out_path, "w") as out_f:
        for fp in files:
            try:
                with open(fp) as f:
                    data = json.load(f)
            except Exception as e:
                print(f"  SKIP {fp}: {e}")
                continue
            n_games += 1

            team_names = data.get("info", {}).get("TeamNames") or []
            if OUR_TEAM_NAME not in team_names:
                continue
            our_idx = team_names.index(OUR_TEAM_NAME)
            rewards = data.get("rewards")
            steps = data.get("steps")
            if not rewards or not steps or len(rewards) != 2:
                continue
            if rewards[our_idx] != 1:
                continue  # safety check -- should already be win-filtered

            used_this_game = False
            for i in range(1, len(steps)):
                p = our_idx
                prev_obs_dict = steps[i - 1][p]["observation"]
                sel = prev_obs_dict.get("select")
                if sel is None:
                    continue
                options = sel.get("option") or []
                if len(options) < 2:
                    continue
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
                        "winner_has_lucario": True,
                        "source": "own_ladder",
                    })
                if not ok:
                    n_errors += 1
                    continue
                for r in rows_this_decision:
                    out_f.write(json.dumps(r) + "\n")
                    n_rows += 1
            if used_this_game:
                n_games_used += 1

    print(f"Games: {n_games}, games used: {n_games_used}, decisions: {n_decisions}, rows: {n_rows}, errors: {n_errors}")
    print(f"FEATURE_DIM = {FEATURE_DIM}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
