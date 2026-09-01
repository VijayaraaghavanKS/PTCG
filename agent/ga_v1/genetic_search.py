"""
Genetic algorithm that evolves the WEIGHTS vector in main.py against real self-play
fitness (win rate over a small local tournament vs our 3 reference sparring
opponents). Standalone script, not part of the submission bundle.

Usage: python genetic_search.py
Writes progress to ga_progress.log and the best-found weights to ga_best_weights.json
(and a full run history to ga_history.json) in this directory.
"""
import os
import sys
import json
import time
import random
import importlib.util

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HARNESS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "harness"))
sys.path.insert(0, HARNESS_DIR)  # for `import cg` (engine) used by run_match

from run_match import load_agent, load_deck, run_match  # noqa: E402

sys.path.insert(0, BASE_DIR)
from main import DEFAULT_WEIGHTS, WEIGHT_KEYS  # noqa: E402

RNG_SEED = 42
random.seed(RNG_SEED)

OPPONENT_DIRS = [
    os.path.abspath(os.path.join(BASE_DIR, "..", "dragapult_day1")),
    os.path.abspath(os.path.join(BASE_DIR, "..", "iono_day1")),
    os.path.abspath(os.path.join(BASE_DIR, "..", "mega_lucario_ref")),
]

# ---------------------------------------------------------------------------
# GA hyperparameters
# ---------------------------------------------------------------------------
POP_SIZE = 24
GENERATIONS = 28
GAMES_PER_OPPONENT = 6      # -> 18 games per candidate per generation (3 opponents)
VALIDATION_GAMES_PER_OPPONENT = 24  # -> 72-game low-noise re-check of top candidates at the end
ELITE_COUNT = 2
TOURNAMENT_K = 3
MUTATION_RATE = 0.25        # probability each gene mutates
MUTATION_SIGMA_FRAC = 0.25  # gaussian sigma as a fraction of the gene's default magnitude
CROSSOVER_RATE = 0.7

TMP_DIR = os.path.join(BASE_DIR, "_ga_tmp")
os.makedirs(TMP_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Load opponents once
# ---------------------------------------------------------------------------
opponents = []
for i, d in enumerate(OPPONENT_DIRS):
    mod = load_agent(f"opp{i}", os.path.join(d, "main.py"), chdir_for_import=d)
    deck = load_deck(os.path.join(d, "deck.csv"))
    opponents.append((os.path.basename(d), mod.agent, deck))


def load_candidate_agent(weights: dict, tag: str):
    """Write weights to a temp file, point GA_WEIGHTS_FILE at it, and import a fresh
    copy of main.py so its module-level WEIGHTS global picks up this candidate."""
    wpath = os.path.join(TMP_DIR, f"weights_{tag}.json")
    with open(wpath, "w") as f:
        json.dump(weights, f)
    os.environ["GA_WEIGHTS_FILE"] = wpath
    spec = importlib.util.spec_from_file_location(f"ga_candidate_{tag}", os.path.join(BASE_DIR, "main.py"))
    mod = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    os.chdir(BASE_DIR)
    try:
        spec.loader.exec_module(mod)
    finally:
        os.chdir(cwd)
    del os.environ["GA_WEIGHTS_FILE"]
    return mod, wpath


def evaluate(weights: dict, tag: str, games_per_opponent: int = GAMES_PER_OPPONENT) -> float:
    mod, wpath = load_candidate_agent(weights, tag)
    my_deck = load_deck(os.path.join(BASE_DIR, "deck.csv"))
    wins = 0
    total = 0
    for opp_name, opp_agent, opp_deck in opponents:
        for g in range(games_per_opponent):
            candidate_first = (g % 2 == 0)
            try:
                if candidate_first:
                    result, _ = run_match(mod.agent, my_deck, opp_agent, opp_deck)
                    win = (result == 0)
                else:
                    result, _ = run_match(opp_agent, opp_deck, mod.agent, my_deck)
                    win = (result == 1)
            except Exception:
                win = False  # crash counts as a loss
            wins += int(win)
            total += 1
    try:
        os.remove(wpath)
    except OSError:
        pass
    return wins / total if total else 0.0


# ---------------------------------------------------------------------------
# GA operators
# ---------------------------------------------------------------------------
def genes_to_weights(genes: list) -> dict:
    return {k: float(v) for k, v in zip(WEIGHT_KEYS, genes)}


def weights_to_genes(w: dict) -> list:
    return [float(w[k]) for k in WEIGHT_KEYS]


def random_individual() -> list:
    genes = []
    for k in WEIGHT_KEYS:
        d = DEFAULT_WEIGHTS[k]
        lo, hi = -1.0 * d, 3.0 * d
        genes.append(random.uniform(lo, hi))
    return genes


def tournament_select(pop, fitnesses, k=TOURNAMENT_K):
    idxs = random.sample(range(len(pop)), k)
    best = max(idxs, key=lambda i: fitnesses[i])
    return pop[best]


def crossover(p1, p2):
    if random.random() > CROSSOVER_RATE:
        return list(p1)
    child = []
    for g1, g2 in zip(p1, p2):
        alpha = random.random()
        child.append(alpha * g1 + (1 - alpha) * g2)
    return child


def mutate(genes):
    out = []
    for k, g in zip(WEIGHT_KEYS, genes):
        if random.random() < MUTATION_RATE:
            sigma = abs(DEFAULT_WEIGHTS[k]) * MUTATION_SIGMA_FRAC
            g = g + random.gauss(0, sigma)
        out.append(g)
    return out


# ---------------------------------------------------------------------------
# Main GA loop
# ---------------------------------------------------------------------------
def main():
    log_path = os.path.join(BASE_DIR, "ga_progress.log")
    history = {"pop_size": POP_SIZE, "generations": GENERATIONS,
               "games_per_candidate": GAMES_PER_OPPONENT * len(opponents),
               "generations_log": []}

    def log(msg):
        print(msg)
        with open(log_path, "a") as f:
            f.write(msg + "\n")

    open(log_path, "w").close()
    t_start = time.time()
    log(f"GA start: pop={POP_SIZE} gens={GENERATIONS} games/candidate={GAMES_PER_OPPONENT*len(opponents)} "
        f"opponents={[o[0] for o in opponents]}")

    population = [weights_to_genes(DEFAULT_WEIGHTS)]
    while len(population) < POP_SIZE:
        population.append(random_individual())

    best_ever_genes = None
    best_ever_fitness = -1.0
    total_games = 0
    # Hall of fame: every generation's #1 candidate, for a low-noise re-check at the end
    # (avoids crowning a winner purely because it got lucky on a small fitness sample).
    hall_of_fame = []  # list of (genes, gen_fitness)

    for gen in range(GENERATIONS):
        t_gen = time.time()
        fitnesses = []
        for i, genes in enumerate(population):
            w = genes_to_weights(genes)
            fit = evaluate(w, tag=f"g{gen}_i{i}")
            fitnesses.append(fit)
            total_games += GAMES_PER_OPPONENT * len(opponents)

        order = sorted(range(len(population)), key=lambda i: fitnesses[i], reverse=True)
        best_idx = order[0]
        gen_best_fit = fitnesses[best_idx]
        gen_avg_fit = sum(fitnesses) / len(fitnesses)
        if gen_best_fit > best_ever_fitness:
            best_ever_fitness = gen_best_fit
            best_ever_genes = list(population[best_idx])
        hall_of_fame.append((list(population[best_idx]), gen_best_fit))

        elapsed = time.time() - t_gen
        total_elapsed = time.time() - t_start
        log(f"gen {gen:02d}: best={gen_best_fit:.3f} avg={gen_avg_fit:.3f} "
            f"best_ever={best_ever_fitness:.3f} gen_time={elapsed:.1f}s total_time={total_elapsed:.1f}s "
            f"total_games={total_games}")
        history["generations_log"].append({
            "gen": gen, "best": gen_best_fit, "avg": gen_avg_fit,
            "best_ever": best_ever_fitness, "gen_time_s": elapsed,
            "total_time_s": total_elapsed, "total_games": total_games,
        })

        # Build next generation
        new_pop = [list(population[i]) for i in order[:ELITE_COUNT]]
        while len(new_pop) < POP_SIZE:
            p1 = tournament_select(population, fitnesses)
            p2 = tournament_select(population, fitnesses)
            child = crossover(p1, p2)
            child = mutate(child)
            new_pop.append(child)
        population = new_pop

    total_time = time.time() - t_start
    log(f"GA evolution done. total_time={total_time:.1f}s total_games={total_games} "
        f"best_ever_fitness(noisy,{GAMES_PER_OPPONENT*len(opponents)}g)={best_ever_fitness:.3f}")

    # ---- Low-noise final validation of top candidates before crowning a winner ----
    # Small per-generation fitness (18 games) is noisy; re-check the strongest distinct
    # candidates seen (hall of fame + overall best-ever + the DEFAULT_WEIGHTS control)
    # with a much bigger sample so the shipped "GA winner" isn't just a lucky draw.
    log(f"Validating top candidates with {VALIDATION_GAMES_PER_OPPONENT*len(opponents)} games each...")
    hof_sorted = sorted(hall_of_fame, key=lambda t: t[1], reverse=True)
    candidates = [("best_ever", best_ever_genes)]
    seen_keys = {tuple(round(g, 6) for g in best_ever_genes)}
    for genes, _ in hof_sorted:
        key = tuple(round(g, 6) for g in genes)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        candidates.append((f"hof_{len(candidates)}", genes))
        if len(candidates) >= 5:
            break
    candidates.append(("default_control", weights_to_genes(DEFAULT_WEIGHTS)))

    val_results = []
    t_val = time.time()
    for name, genes in candidates:
        w = genes_to_weights(genes)
        fit = evaluate(w, tag=f"val_{name}", games_per_opponent=VALIDATION_GAMES_PER_OPPONENT)
        val_results.append((name, fit, genes))
        log(f"  validation[{name}]: fitness={fit:.3f} over {VALIDATION_GAMES_PER_OPPONENT*len(opponents)} games")
    val_time = time.time() - t_val

    val_results.sort(key=lambda t: t[1], reverse=True)
    winner_name, winner_fit, winner_genes = val_results[0]
    log(f"Validation winner: {winner_name} fitness={winner_fit:.3f} (validation took {val_time:.1f}s)")

    best_weights = genes_to_weights(winner_genes)
    with open(os.path.join(BASE_DIR, "ga_best_weights.json"), "w") as f:
        json.dump(best_weights, f, indent=2)
    history["validation"] = {
        "games_per_opponent": VALIDATION_GAMES_PER_OPPONENT,
        "results": [{"name": n, "fitness": f} for n, f, _ in val_results],
        "winner": winner_name,
        "validation_time_s": val_time,
        "total_time_s": time.time() - t_start,
    }
    with open(os.path.join(BASE_DIR, "ga_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    log(f"Best weights (post-validation, winner={winner_name}) written to ga_best_weights.json: {best_weights}")
    log(f"GA + validation total_time={time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
