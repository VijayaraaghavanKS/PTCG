"""
Extract (feature, label, group) rows from self-play win-recordings produced by
selfplay_record.py, using bc_v2's exact feature extractor (imported unmodified).
Output schema matches bc_v2/extracted_rows.jsonl so the two can be concatenated.

Run from repo root with the project venv active:
  python3 agent/bc_dragapult_v3_combined/extract_selfplay.py <selfplay_dir> <out_jsonl>
"""
import glob
import json
import os
import sys

BC_V2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "bc_v2")
sys.path.insert(0, BC_V2)

from cg.api import all_card_data, all_attack, to_observation_class  # noqa: E402
from features import extract_features, FEATURE_DIM  # noqa: E402


def main():
    selfplay_dir = sys.argv[1]
    out_path = sys.argv[2]

    all_card = all_card_data()
    card_table = {c.cardId: c for c in all_card}
    all_atk = all_attack()
    attack_table = {a.attackId: a for a in all_atk}

    files = sorted(glob.glob(os.path.join(selfplay_dir, "*.json")))
    print(f"Found {len(files)} self-play game files")

    n_games = 0
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
            for dec in data.get("decisions", []):
                obs_dict = dec["observation"]
                chosen = dec["action"]
                sel = obs_dict.get("select")
                if sel is None:
                    continue
                options = sel.get("option") or []
                if len(options) < 2:
                    continue
                chosen_set = set(chosen)
                if not chosen_set or not all(0 <= c < len(options) for c in chosen_set):
                    continue
                try:
                    obs = to_observation_class(obs_dict)
                except Exception:
                    n_errors += 1
                    continue
                if obs.select is None or obs.current is None:
                    continue

                group_counter += 1
                n_decisions += 1
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
                        "winner_has_dragapult": True,
                        "source": "selfplay",
                    })
                if not ok:
                    n_errors += 1
                    continue
                for r in rows_this_decision:
                    out_f.write(json.dumps(r) + "\n")
                    n_rows += 1

    print(f"Games: {n_games}, decisions: {n_decisions}, rows: {n_rows}, errors: {n_errors}")
    print(f"FEATURE_DIM = {FEATURE_DIM}")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
