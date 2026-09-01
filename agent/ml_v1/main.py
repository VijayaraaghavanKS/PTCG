import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cg.api import Observation, all_card_data, all_attack, to_observation_class
from features import extract_features

"""
ml_v1: a prototype agent whose option-scoring is a small logistic-regression
model trained via imitation learning on real Kaggle top-episode replays,
instead of hand-tuned heuristic weights.

For every legal Option in obs.select.option, we build a generic (deck-agnostic)
feature vector (see features.py) describing the select context, the option
type, and the stats of whatever card/Pokemon the option refers to (HP, energy
count, attack damage, prize value, etc. -- never raw card IDs, since the
training replays used decks with mostly different card pools than our own).
The model scores each option; we act on the highest-scoring ones.

No third-party runtime dependencies (stdlib only), so it is safe to run
inside the Kaggle submission sandbox regardless of what's installed there.
"""

# Load deck.csv in the dataset
file_path = "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/" + file_path
with open(file_path, "r") as file:
    csv = file.read().split("\n")
my_deck = []
for i in range(60):
    my_deck.append(int(csv[i]))

# Card / attack metadata tables (global, deck-independent)
all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}
all_atk = all_attack()
attack_table = {a.attackId: a for a in all_atk}

# Learned linear scoring weights (trained offline via features.py + train.py
# on ~60 real top-episode replays; see research notes)
weights_path = "weights.json"
if not os.path.exists(weights_path):
    weights_path = "/kaggle_simulations/agent/" + weights_path
with open(weights_path, "r") as f:
    _model = json.load(f)
COEF = _model["coef"]
INTERCEPT = _model["intercept"]


def score_option(obs: Observation, option) -> float:
    feats = extract_features(obs, option, card_table, attack_table)
    s = INTERCEPT
    for w, x in zip(COEF, feats):
        if x != 0.0:
            s += w * x
    return s


def agent(obs_dict: dict) -> list[int]:
    """Main Agent Function.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount
    (inclusive), with no duplicate elements.
    """
    obs = to_observation_class(obs_dict)
    if obs.select is None:
        # Initial deck submission.
        return my_deck

    select = obs.select
    options = select.option

    scores = [score_option(obs, o) for o in options]
    desc_indices = [i for i, _ in sorted(enumerate(scores), key=lambda x: x[1], reverse=True)]

    max_c = select.maxCount if select.maxCount is not None else 1

    # Mirror the simple "take the top-maxCount scoring options" convention used
    # by the reference heuristic agents in this repo -- this is a known-good,
    # simple way to handle optional/multi-select without extra special-casing.
    k = max(min(max_c, len(options)), 0)
    return desc_indices[:k]
