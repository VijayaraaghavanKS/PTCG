import sys
sys.path.insert(0, "/Users/vijayaraaghavanks/Dev/PTCG/research")
from ab_test import run_matchups  # noqa

ROOT = "/Users/vijayaraaghavanks/Dev/PTCG/agent"
NEW = f"{ROOT}/lucario_crossplay_reranker_v1"
OLDRERANKER = f"{ROOT}/lucario_reranker_v1"
NORERANKER = f"{ROOT}/lucario_deck_tune_v1"

N = 300
matchups = [
    ("CONFIRM_NEW_vs_iono", NEW, f"{ROOT}/iono_day1", N),
    ("CONFIRM_OLDRERANKER_vs_iono", OLDRERANKER, f"{ROOT}/iono_day1", N),
    ("CONFIRM_NORERANKER_vs_iono", NORERANKER, f"{ROOT}/iono_day1", N),
    ("CONFIRM_NEW_vs_crustle", NEW, f"{ROOT}/crustle_wall_ref", N),
    ("CONFIRM_OLDRERANKER_vs_crustle", OLDRERANKER, f"{ROOT}/crustle_wall_ref", N),
    ("CONFIRM_NORERANKER_vs_crustle", NORERANKER, f"{ROOT}/crustle_wall_ref", N),
]

if __name__ == "__main__":
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    run_matchups(matchups, n_workers)
