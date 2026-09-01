"""
Re-extract the same real Dragapult-specific win-filtered games used by
agent/bc_dragapult_v3_combined (public top-episode-index Dragapult wins +
our own live-ladder Dragapult wins), but with the TurnContextTracker's 6
extra "already committed this turn" features appended (144-dim total vs the
prior 138-dim). See features.py docstring for the leak-safety argument.

Source A: research/episodes/**/*.json, winner's deck contains Dragapult ex (id 121).
Source B: research/own_replays/*.json, our own team name, win-filtered (rewards[our_idx]==1).

Usage: python3 extract_plan_data.py <out_public.jsonl> <out_own.jsonl>
"""
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BC_V2 = os.path.join(HERE, "..", "bc_v2")
sys.path.insert(0, BC_V2)  # for cg package
sys.path.insert(0, HERE)  # must win over bc_v2's own features.py

from cg.api import all_card_data, all_attack, to_observation_class  # noqa: E402
from features import extract_features, FEATURE_DIM, TurnContextTracker  # noqa: E402

RESEARCH_EPISODES = os.path.join(HERE, "..", "..", "research", "episodes")
OWN_REPLAYS = os.path.join(HERE, "..", "..", "research", "own_replays")
OUR_TEAM_NAME = "K S VIJAYARAAGHAVAN"
DRAGAPULT_EX = 121


def process_game(fp, data, our_idx, card_table, attack_table, source_tag, group_start):
    """Walk one game's steps in chronological order, maintaining a
    TurnContextTracker, emitting rows for the target player's non-trivial
    decisions. Returns (rows, next_group_counter)."""
    steps = data.get("steps")
    if not steps:
        return [], group_start

    tracker = TurnContextTracker()
    rows = []
    group_counter = group_start
    p = our_idx

    for i in range(1, len(steps)):
        prev_obs_dict = steps[i - 1][p]["observation"]
        sel = prev_obs_dict.get("select")
        if sel is None:
            continue
        options = sel.get("option") or []
        chosen = steps[i][p]["action"]
        if chosen is None:
            continue
        chosen_set = set(chosen)
        if not chosen_set or not all(0 <= c < len(options) for c in chosen_set):
            continue

        try:
            obs = to_observation_class(prev_obs_dict)
        except Exception:
            continue
        if obs.select is None or obs.current is None:
            continue

        # Only emit a training row (with plan-context features) for
        # non-trivial (>=2 option) decisions -- same filter as the base
        # extractor. Trivial/forced decisions still update the tracker below
        # (they're real actions that happened), just don't get a row.
        if len(options) >= 2:
            group_counter += 1
            ok = True
            rows_this_decision = []
            for oi, opt in enumerate(obs.select.option):
                try:
                    feat = extract_features(obs, opt, card_table, attack_table, tracker)
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
                    "source": source_tag,
                })
            if ok:
                rows.extend(rows_this_decision)

        # Advance the tracker using the ACTUAL chosen option(s), regardless
        # of whether this decision was trivial -- real turn history either
        # way. Causal: only uses info already known at this point in the replay.
        try:
            for oi in sorted(chosen_set):
                if oi < len(obs.select.option):
                    tracker.observe_chosen(obs, obs.select.option[oi])
        except Exception:
            pass

    return rows, group_counter


def extract_public(out_path, card_table, attack_table):
    pattern = os.path.join(RESEARCH_EPISODES, "**", "*.json")
    files = sorted(glob.glob(pattern, recursive=True))
    print(f"[public] found {len(files)} files")
    n_games = 0
    n_used = 0
    n_rows = 0
    group_counter = 0
    with open(out_path, "w") as out_f:
        for fp in files:
            try:
                with open(fp) as f:
                    data = json.load(f)
            except Exception:
                continue
            n_games += 1
            rewards = data.get("rewards")
            steps = data.get("steps")
            if not rewards or not steps or len(rewards) != 2 or rewards[0] == rewards[1]:
                continue
            winner_idx = 0 if rewards[0] > rewards[1] else 1
            try:
                step0_viz = steps[0][0]["visualize"][0]["action"]
                winner_deck = step0_viz[winner_idx]
                if DRAGAPULT_EX not in winner_deck:
                    continue
            except Exception:
                continue
            rows, group_counter = process_game(fp, data, winner_idx, card_table, attack_table, "public", group_counter)
            if rows:
                n_used += 1
                for r in rows:
                    out_f.write(json.dumps(r) + "\n")
                    n_rows += 1
    print(f"[public] games_used={n_used} rows={n_rows}")
    return n_rows


def extract_own(out_path, card_table, attack_table):
    files = sorted(glob.glob(os.path.join(OWN_REPLAYS, "*.json")))
    print(f"[own] found {len(files)} files")
    n_used = 0
    n_rows = 0
    group_counter = 0
    with open(out_path, "w") as out_f:
        for fp in files:
            try:
                with open(fp) as f:
                    data = json.load(f)
            except Exception:
                continue
            team_names = data.get("info", {}).get("TeamNames") or []
            if OUR_TEAM_NAME not in team_names:
                continue
            our_idx = team_names.index(OUR_TEAM_NAME)
            rewards = data.get("rewards")
            if not rewards or len(rewards) != 2 or rewards[our_idx] != 1:
                continue
            rows, group_counter = process_game(fp, data, our_idx, card_table, attack_table, "own", group_counter)
            if rows:
                n_used += 1
                for r in rows:
                    out_f.write(json.dumps(r) + "\n")
                    n_rows += 1
    print(f"[own] games_used={n_used} rows={n_rows}")
    return n_rows


def main():
    out_public = sys.argv[1]
    out_own = sys.argv[2]
    all_card = all_card_data()
    card_table = {c.cardId: c for c in all_card}
    attack_table = {a.attackId: a for a in all_attack()}
    print(f"FEATURE_DIM = {FEATURE_DIM}")
    extract_public(out_public, card_table, attack_table)
    extract_own(out_own, card_table, attack_table)


if __name__ == "__main__":
    main()
