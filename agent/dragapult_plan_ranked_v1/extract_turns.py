"""Extract turn-level (feature, label) rows from real replay JSONs for the
turn-level plan-ranking experiment (dragapult_plan_ranked_v1).

For each of "our" turns in a winning game, build:
  - features describing the state BEFORE the turn started + what we did
    during the turn (prize/hp deltas, energy/evolve/trainer counts, attacked
    flag) -- the same primitives dragapult_plan_v1's hand-tuned rollout
    scorer already uses.
  - a label = this-turn's immediate prize/hp delta PLUS the following
    same-side turn's delta (a 2-turn lookahead bootstrap), so the learned
    model targets something the myopic single-turn hand-tuned score cannot
    see: whether a turn set up a stronger follow-up, not just its own
    immediate payoff.

Turn segmentation: within a single player's own observation stream,
`current.turn` only advances on that player's own successive turns (verified
empirically: player0's stream shows turn values 0,1,3,5,7... i.e. each
distinct value covers exactly one of that player's own turns, opponent turns
interleaved invisibly). Group consecutive steps by this player's own
`current.turn` value.
"""
import glob
import json
import os
import sys

FEATURE_KEYS = [
    "pre_my_hp", "pre_op_hp", "pre_my_prize_left", "pre_op_prize_left",
    "pre_my_bench_n", "pre_op_bench_n", "pre_my_energy_total", "turn_num",
    "attacked", "n_evolves", "n_attaches", "n_trainers", "n_ability",
]


def hp_total(players_side):
    total = 0
    for p in list(players_side.get("active") or []) + list(players_side.get("bench") or []):
        if p:
            total += p.get("hp", 0)
    return total


def prize_left(side):
    return sum(1 for c in (side.get("prize") or []) if c is None)


def energy_total(side):
    total = 0
    for p in list(side.get("active") or []) + list(side.get("bench") or []):
        if p:
            total += len(p.get("energies") or [])
    return total


def load_game(fp):
    with open(fp) as f:
        return json.load(f)


def find_my_idx(d, name_substr=None):
    names = d.get("info", {}).get("TeamNames", [])
    if name_substr:
        for i, n in enumerate(names):
            if n and name_substr.upper() in n.upper():
                return i
        return None
    # for public/generic games: caller passes explicit idx externally
    return None


def extract_turns_for_side(d, my_idx):
    """Return list of dicts: {turn_num, feat(list), delta(this-turn), attacked...}"""
    steps = d["steps"]
    segments = []  # list of (turn_value, [step_indices])
    cur_turn_val = None
    cur_seg = []
    for i, step in enumerate(steps):
        try:
            obs = step[my_idx]["observation"]
        except Exception:
            continue
        cur = obs.get("current")
        if not isinstance(cur, dict):
            continue
        if cur.get("yourIndex") != my_idx:
            continue
        tv = cur.get("turn")
        if tv is None:
            continue
        if cur_turn_val is None:
            cur_turn_val = tv
        if tv != cur_turn_val:
            segments.append((cur_turn_val, cur_seg))
            cur_seg = []
            cur_turn_val = tv
        cur_seg.append(i)
    if cur_seg:
        segments.append((cur_turn_val, cur_seg))

    op_idx = 1 - my_idx
    rows = []
    for seg_i, (tv, idxs) in enumerate(segments):
        if len(idxs) < 1:
            continue
        first_step = steps[idxs[0]][my_idx]["observation"]
        last_step = steps[idxs[-1]][my_idx]["observation"]
        pre_cur = first_step.get("current")
        post_cur = last_step.get("current")
        if not isinstance(pre_cur, dict) or not isinstance(post_cur, dict):
            continue
        try:
            pre_me = pre_cur["players"][my_idx]
            pre_op = pre_cur["players"][op_idx]
            post_me = post_cur["players"][my_idx]
            post_op = post_cur["players"][op_idx]
        except Exception:
            continue

        pre_my_hp = hp_total(pre_me)
        pre_op_hp = hp_total(pre_op)
        post_my_hp = hp_total(post_me)
        post_op_hp = hp_total(post_op)
        pre_my_prize = prize_left(pre_me)
        pre_op_prize = prize_left(pre_op)
        post_my_prize = prize_left(post_me)
        post_op_prize = prize_left(post_op)

        prizes_taken = max(0, pre_op_prize - post_op_prize)
        prizes_lost = max(0, pre_my_prize - post_my_prize)
        hp_swing = (post_op_hp - pre_op_hp) * -1 + (post_my_hp - pre_my_hp)
        # positive hp_swing = we dealt more damage than we took (roughly)
        hp_swing = (pre_op_hp - post_op_hp) - (pre_my_hp - post_my_hp)

        attacked = 0
        n_evolves = 0
        n_attaches = 0
        n_trainers = 0
        n_ability = 0
        for si in idxs:
            try:
                o = steps[si][my_idx]["action"]
            except Exception:
                o = None
            sel = steps[si][my_idx]["observation"].get("select")
            if not isinstance(sel, dict) or not sel.get("option"):
                continue
            opts = sel["option"]
            if not isinstance(o, list) or not o:
                continue
            pick = o[0] if isinstance(o[0], int) else None
            if pick is None or pick >= len(opts):
                continue
            otype = opts[pick].get("type") if isinstance(opts[pick], dict) else None
            # OptionType (cg/api.py): PLAY=7, ATTACH=8, EVOLVE=9, ABILITY=10, ATTACK=13
            if otype == 13:
                attacked = 1
            elif otype == 9:
                n_evolves += 1
            elif otype == 8:
                n_attaches += 1
            elif otype == 7:
                n_trainers += 1
            elif otype == 10:
                n_ability += 1

        # POST-turn features: state AFTER this turn's rollout completed. This
        # is what's actually available at candidate-scoring time inside
        # compute_turn_plan (the rollout produces a resulting state `cs`,
        # never the pre-turn state) -- training on pre-turn features would
        # not match how the model is queried at inference.
        post_feat = [
            float(post_my_hp), float(post_op_hp), float(post_my_prize), float(post_op_prize),
            float(len(post_me.get("bench") or [])), float(len(post_op.get("bench") or [])),
            float(energy_total(post_me)), float(tv if tv is not None else 0),
            float(attacked), float(n_evolves), float(n_attaches), float(n_trainers), float(n_ability),
        ]
        this_turn_value = prizes_taken * 3.0 - prizes_lost * 3.0 + hp_swing / 50.0
        rows.append({
            "seg_i": seg_i, "turn": tv, "feat": post_feat,
            "this_turn_value": this_turn_value,
        })

    # Bootstrapped 1-step state-value label: given the state immediately
    # AFTER this turn (= what a candidate branch's rollout produces), the
    # label is how much value the FOLLOWING same-side turn achieved. This is
    # the piece a myopic single-turn score cannot see: whether the resulting
    # position sets up a strong follow-up turn, not just this turn's own
    # payoff (which the hand-tuned formula already measures directly).
    for i, r in enumerate(rows):
        r["label"] = rows[i + 1]["this_turn_value"] if i + 1 < len(rows) else 0.0
    return rows


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "turn_rows.jsonl"
    own_dir = "research/own_replays"
    n_games = 0
    n_rows = 0
    with open(out_path, "w") as out:
        for fp in sorted(glob.glob(os.path.join(own_dir, "*.json"))):
            try:
                d = load_game(fp)
            except Exception:
                continue
            my_idx = find_my_idx(d, "VIJAYARAAGHAVAN")
            if my_idx is None:
                continue
            rewards = d.get("rewards", [0, 0])
            if my_idx >= len(rewards) or rewards[my_idx] != 1:
                continue  # win-filtered
            rows = extract_turns_for_side(d, my_idx)
            if not rows:
                continue
            n_games += 1
            for r in rows:
                r["source"] = "own"
                r["game"] = os.path.basename(fp)
                out.write(json.dumps(r) + "\n")
                n_rows += 1
    print(f"own_replays: {n_games} win-games, {n_rows} turn rows")


if __name__ == "__main__":
    main()
