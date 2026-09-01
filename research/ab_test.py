"""Parallel A/B match runner: given a list of (agent_dir_A, agent_dir_B, n_games)
matchups, run them all in parallel across CPU cores and report win rates.
Alternates which physical slot each side plays, matching this project's
established no-positional-bias convention.

Usage:
  python3 research/ab_test.py <n_workers>
  (matchup list is defined in MATCHUPS below -- edited per invocation)
"""
import itertools
import multiprocessing as mp
import os
import sys
import time

HARNESS = "/Users/vijayaraaghavanks/Dev/PTCG/agent/harness"
sys.path.insert(0, HARNESS)

_W = {}


def worker_init():
    from run_match import load_agent, load_deck, run_match  # noqa
    _W["load_agent"] = load_agent
    _W["load_deck"] = load_deck
    _W["run_match"] = run_match
    _W["cache"] = {}


def _get(dir_path):
    cache = _W["cache"]
    if dir_path not in cache:
        name = os.path.basename(dir_path.rstrip("/"))
        mod = _W["load_agent"](f"a_{name}_{os.getpid()}", os.path.join(dir_path, "main.py"), chdir_for_import=dir_path)
        deck = _W["load_deck"](os.path.join(dir_path, "deck.csv"))
        cache[dir_path] = (mod.agent, deck)
    return cache[dir_path]


def play(task):
    dir_a, dir_b, game_idx, tag = task
    agent_a, deck_a = _get(dir_a)
    agent_b, deck_b = _get(dir_b)
    run_match = _W["run_match"]
    try:
        if game_idx % 2 == 0:
            result, sel = run_match(agent_a, deck_a, agent_b, deck_b)
            a_won = result == 0
        else:
            result, sel = run_match(agent_b, deck_b, agent_a, deck_a)
            a_won = result == 1
        if result == 2:
            return (tag, "draw")
        return (tag, "a" if a_won else "b")
    except Exception as e:
        return (tag, f"error:{type(e).__name__}:{e}")


def run_matchups(matchups, n_workers):
    """matchups: list of (tag, dir_a, dir_b, n_games)"""
    tasks = []
    for tag, dir_a, dir_b, n_games in matchups:
        for i in range(n_games):
            tasks.append((dir_a, dir_b, i, tag))
    t0 = time.time()
    with mp.Pool(n_workers, initializer=worker_init) as pool:
        results = pool.map(play, tasks, chunksize=8)
    dt = time.time() - t0

    tally = {tag: {"a": 0, "b": 0, "draw": 0, "error": 0} for tag, _, _, _ in matchups}
    for tag, outcome in results:
        if outcome.startswith("error"):
            tally[tag]["error"] += 1
            print(f"  ERROR [{tag}]: {outcome}")
        else:
            tally[tag][outcome] += 1

    print(f"\n=== A/B results ({dt:.1f}s, {len(tasks)} games, {len(tasks)/dt:.1f} games/s) ===")
    for tag, dir_a, dir_b, n_games in matchups:
        t = tally[tag]
        total = t["a"] + t["b"] + t["draw"]
        a_pct = 100.0 * t["a"] / total if total else 0.0
        print(f"{tag:40s} A({os.path.basename(dir_a)})={t['a']}/{total} ({a_pct:.1f}%)  "
              f"B({os.path.basename(dir_b)})={t['b']}  draw={t['draw']}  err={t['error']}")
    return tally


if __name__ == "__main__":
    print("This module is meant to be imported; see research/run_crossplay_ab.py")
