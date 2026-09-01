"""
Cross-play data generation for the Mega Lucario ex line: run games between
DIFFERENT pairs of the 4 legitimate local Lucario-line variants (not just
mirrors), full CPU parallelism, extract decision-level BC training rows from
the WINNING side of each game only. Direct adaptation of
research/crossplay_gen.py (the Dragapult version -- see AGENT_LOG.md's
"Cross-play data generation revisited" entry) for the much smaller Lucario
pool: 4 variants -> C(4,2)=6 cross-pairs instead of Dragapult's 78, so this
run compensates with higher N per pair rather than more pairs.

Pool: mega_lucario_ref (base heuristic), mega_lucario_v2 (belief+search),
lucario_deck_tune_v1 (Gravity Mountain bugfix), lucario_reranker_v1 (current
live BC reranker, first-gen model). All 4 confirmed to share the identical
60-card decklist (byte-identical deck.csv), so this is genuinely a policy
cross-play, not a deck-composition confound.

Extracts features directly in-process (no raw per-game JSON ever written to
disk), reusing lucario_reranker_v1/features.py (byte-identical to
bc_v2/dragapult_combined_final_v2's 138-dim extractor). Each worker writes
its own row shard (rows_shard_<n>.jsonl).

Usage:
  python3 research/crossplay_gen_lucario.py <out_dir> <games_per_pair> <mirror_games_per_variant> [n_workers]
"""
import itertools
import json
import multiprocessing as mp
import os
import random
import sys
import time

ROOT = "/Users/vijayaraaghavanks/Dev/PTCG"
HARNESS = os.path.join(ROOT, "agent", "harness")
FEAT_DIR = os.path.join(ROOT, "agent", "lucario_reranker_v1")  # canonical 138-dim features.py

POOL_NAMES = [
    "mega_lucario_ref", "mega_lucario_v2", "lucario_deck_tune_v1", "lucario_reranker_v1",
]
POOL_PATHS = {name: os.path.join(ROOT, "agent", name) for name in POOL_NAMES}

MAX_SELECTS = 5000

_W = {}  # per-worker globals


def worker_init(out_dir, worker_id):
    sys.path.insert(0, HARNESS)
    from run_match import load_agent, load_deck  # noqa
    from cg.game import battle_start, battle_select, battle_finish  # noqa
    from cg.api import all_card_data, all_attack, to_observation_class  # noqa

    sys.path.insert(0, FEAT_DIR)
    import features as feat_mod  # noqa

    _W["battle_start"] = battle_start
    _W["battle_select"] = battle_select
    _W["battle_finish"] = battle_finish
    _W["to_obs"] = to_observation_class
    _W["extract_features"] = feat_mod.extract_features
    _W["card_table"] = {c.cardId: c for c in all_card_data()}
    _W["attack_table"] = {a.attackId: a for a in all_attack()}

    agents = {}
    for name, path in POOL_PATHS.items():
        mod = load_agent(f"w{worker_id}_{name}", os.path.join(path, "main.py"), chdir_for_import=path)
        deck = load_deck(os.path.join(path, "deck.csv"))
        agents[name] = (mod.agent, deck)
    _W["agents"] = agents
    _W["group_ctr"] = itertools.count(1)
    _W["out_f"] = open(os.path.join(out_dir, f"rows_shard_{worker_id}.jsonl"), "w")
    _W["worker_id"] = worker_id


def _extract_rows(decisions, winner_name, opp_name, game_key):
    to_obs = _W["to_obs"]
    extract_features = _W["extract_features"]
    card_table = _W["card_table"]
    attack_table = _W["attack_table"]
    out_f = _W["out_f"]
    n_rows = 0
    for obs_dict, chosen in decisions:
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
            obs_c = to_obs(obs_dict)
        except Exception:
            continue
        if obs_c.select is None or obs_c.current is None:
            continue
        group_id = next(_W["group_ctr"])
        rows_this = []
        ok = True
        for oi, opt in enumerate(obs_c.select.option):
            try:
                feat = extract_features(obs_c, opt, card_table, attack_table)
            except Exception:
                ok = False
                break
            label = 1 if oi in chosen_set else 0
            rows_this.append({
                "group": group_id,
                "label": label,
                "feat": feat,
                "n_options": len(options),
                "winner_variant": winner_name,
                "opponent_variant": opp_name,
                "source": "crossplay",
                "game_key": game_key,
            })
        if not ok:
            continue
        for r in rows_this:
            out_f.write(json.dumps(r) + "\n")
            n_rows += 1
    out_f.flush()
    return n_rows


def play_one(task):
    var_a, var_b, game_idx = task
    agent_a, deck_a = _W["agents"][var_a]
    agent_b, deck_b = _W["agents"][var_b]
    if game_idx % 2 == 0:
        s0n, s0a, s0d = var_a, agent_a, deck_a
        s1n, s1a, s1d = var_b, agent_b, deck_b
    else:
        s0n, s0a, s0d = var_b, agent_b, deck_b
        s1n, s1a, s1d = var_a, agent_a, deck_a

    battle_start = _W["battle_start"]
    battle_select = _W["battle_select"]
    battle_finish = _W["battle_finish"]

    try:
        obs, start = battle_start(s0d, s1d)
        if start.errorPlayer >= 0:
            return (var_a, var_b, None, 0, "deck_error")
        dec0, dec1 = [], []
        selects = 0
        while obs["current"]["result"] < 0:
            yi = obs["current"]["yourIndex"]
            sel = obs.get("select")
            n_opts = len(sel["option"]) if sel and sel.get("option") else 0
            if yi == 0:
                choice = s0a(obs)
                if n_opts >= 2:
                    dec0.append((obs, list(choice)))
            else:
                choice = s1a(obs)
                if n_opts >= 2:
                    dec1.append((obs, list(choice)))
            obs = battle_select(choice)
            selects += 1
            if selects > MAX_SELECTS:
                battle_finish()
                return (var_a, var_b, None, 0, "max_selects")
        result = obs["current"]["result"]
        battle_finish()
    except Exception as e:
        return (var_a, var_b, None, 0, f"error:{type(e).__name__}:{e}")

    if result == 2:
        return (var_a, var_b, "draw", 0, None)

    winner_name = s0n if result == 0 else s1n
    winner_decisions = dec0 if result == 0 else dec1
    opp_name = var_b if winner_name == var_a else var_a
    game_key = f"{var_a}__{var_b}__{game_idx}"
    n_rows = _extract_rows(winner_decisions, winner_name, opp_name, game_key)
    return (var_a, var_b, winner_name, n_rows, None)


def build_tasks(games_per_pair, mirror_games_per_variant, seed=1234):
    rng = random.Random(seed)
    tasks = []
    cross_pairs = list(itertools.combinations(POOL_NAMES, 2))
    for a, b in cross_pairs:
        for i in range(games_per_pair):
            tasks.append((a, b, i))
    for a in POOL_NAMES:
        for i in range(mirror_games_per_variant):
            tasks.append((a, a, i))
    rng.shuffle(tasks)
    return tasks, len(cross_pairs)


def worker_entry(worker_id, out_dir, task_chunk, result_q):
    worker_init(out_dir, worker_id)
    local_win = {}
    local_err = {}
    local_rows = 0
    local_draws = 0
    local_games = 0
    for task in task_chunk:
        var_a, var_b, winner, n_rows, err = play_one(task)
        local_games += 1
        if err:
            local_err[err.split(":")[0]] = local_err.get(err.split(":")[0], 0) + 1
        elif winner == "draw":
            local_draws += 1
        else:
            local_win[winner] = local_win.get(winner, 0) + 1
            local_rows += n_rows
        if local_games % 500 == 0:
            print(f"  [worker {worker_id}] {local_games}/{len(task_chunk)} games, rows={local_rows}", flush=True)
    _W["out_f"].close()
    result_q.put((worker_id, local_win, local_err, local_rows, local_draws, local_games))


def main2():
    out_dir = sys.argv[1]
    games_per_pair = int(sys.argv[2])
    mirror_games_per_variant = int(sys.argv[3])
    n_workers = int(sys.argv[4]) if len(sys.argv) > 4 else (os.cpu_count() or 8)
    os.makedirs(out_dir, exist_ok=True)

    tasks, n_cross_pairs = build_tasks(games_per_pair, mirror_games_per_variant)
    print(f"Pool size: {len(POOL_NAMES)}, cross-pairs: {n_cross_pairs}, "
          f"total tasks: {len(tasks)} (cross={n_cross_pairs*games_per_pair}, "
          f"mirror={len(POOL_NAMES)*mirror_games_per_variant})", flush=True)
    print(f"Workers: {n_workers}", flush=True)

    chunks = [tasks[i::n_workers] for i in range(n_workers)]
    result_q = mp.Queue()
    procs = []
    t0 = time.time()
    for wid, chunk in enumerate(chunks):
        p = mp.Process(target=worker_entry, args=(wid, out_dir, chunk, result_q))
        p.start()
        procs.append(p)

    win_tally, err_tally = {}, {}
    total_rows, total_draws, total_games = 0, 0, 0
    for _ in procs:
        wid, lw, le, lr, ld, lg = result_q.get()
        for k, v in lw.items():
            win_tally[k] = win_tally.get(k, 0) + v
        for k, v in le.items():
            err_tally[k] = err_tally.get(k, 0) + v
        total_rows += lr
        total_draws += ld
        total_games += lg
        print(f"worker {wid} done: games={lg} rows={lr} draws={ld} errs={le}", flush=True)

    for p in procs:
        p.join()

    dt = time.time() - t0
    print(f"\n=== DONE in {dt:.1f}s ({total_games/dt:.1f} games/s) ===")
    print(f"Total games: {total_games}, draws: {total_draws}, total rows: {total_rows}")
    print(f"Errors: {err_tally}")
    print("Win tally by variant:")
    for k, v in sorted(win_tally.items(), key=lambda kv: -kv[1]):
        print(f"  {k:35s} {v}")
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({
            "total_games": total_games, "total_rows": total_rows, "draws": total_draws,
            "errors": err_tally, "win_tally": win_tally, "wall_clock_s": dt,
            "games_per_pair": games_per_pair, "mirror_games_per_variant": mirror_games_per_variant,
            "n_workers": n_workers, "pool": POOL_NAMES,
        }, f, indent=2)


if __name__ == "__main__":
    main2()
