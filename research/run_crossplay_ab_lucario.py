import sys
sys.path.insert(0, "/Users/vijayaraaghavanks/Dev/PTCG/research")
from ab_test import run_matchups  # noqa

ROOT = "/Users/vijayaraaghavanks/Dev/PTCG/agent"
NEW = f"{ROOT}/lucario_crossplay_reranker_v1"
BASELINE_RERANKER = f"{ROOT}/lucario_reranker_v1"
BASE_NO_RERANKER = f"{ROOT}/lucario_deck_tune_v1"

REFS_STD = ["dragapult_day1", "iono_day1", "mega_lucario_ref", "crustle_wall_ref"]
REFS_NEW = ["festival_dipplin_ref", "okidogi_bruiser_ref", "honchkrow_control_ref", "grass_box_ref"]

N_STD = 250
N_NEW = 190
N_ISO = 250

matchups = [
    ("isolation_new_vs_oldreranker", NEW, BASELINE_RERANKER, N_ISO),
    ("isolation_new_vs_noreranker", NEW, BASE_NO_RERANKER, N_ISO),
]
for r in REFS_STD:
    matchups.append((f"NEW_vs_{r}", NEW, f"{ROOT}/{r}", N_STD))
    matchups.append((f"OLDRERANKER_vs_{r}", BASELINE_RERANKER, f"{ROOT}/{r}", N_STD))
    matchups.append((f"NORERANKER_vs_{r}", BASE_NO_RERANKER, f"{ROOT}/{r}", N_STD))
for r in REFS_NEW:
    matchups.append((f"NEW_vs_{r}", NEW, f"{ROOT}/{r}", N_NEW))
    matchups.append((f"OLDRERANKER_vs_{r}", BASELINE_RERANKER, f"{ROOT}/{r}", N_NEW))
    matchups.append((f"NORERANKER_vs_{r}", BASE_NO_RERANKER, f"{ROOT}/{r}", N_NEW))

if __name__ == "__main__":
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    run_matchups(matchups, n_workers)
