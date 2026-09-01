import sys
sys.path.insert(0, "/Users/vijayaraaghavanks/Dev/PTCG/research")
from ab_test import run_matchups  # noqa

ROOT = "/Users/vijayaraaghavanks/Dev/PTCG/agent"
NEW = f"{ROOT}/dragapult_crossplay_reranker_v1"
BASE = f"{ROOT}/dragapult_combined_final_v2"

REFS_STD = ["dragapult_day1", "iono_day1", "mega_lucario_ref", "crustle_wall_ref"]
REFS_NEW = ["festival_dipplin_ref", "okidogi_bruiser_ref", "honchkrow_control_ref", "grass_box_ref"]

N_STD = 250
N_NEW = 190
N_ISO = 250

matchups = [("isolation_new_vs_base", NEW, BASE, N_ISO)]
for r in REFS_STD:
    matchups.append((f"NEW_vs_{r}", NEW, f"{ROOT}/{r}", N_STD))
    matchups.append((f"BASE_vs_{r}", BASE, f"{ROOT}/{r}", N_STD))
for r in REFS_NEW:
    matchups.append((f"NEW_vs_{r}", NEW, f"{ROOT}/{r}", N_NEW))
    matchups.append((f"BASE_vs_{r}", BASE, f"{ROOT}/{r}", N_NEW))

if __name__ == "__main__":
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    run_matchups(matchups, n_workers)
