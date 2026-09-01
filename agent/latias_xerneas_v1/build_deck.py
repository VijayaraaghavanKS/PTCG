"""Builds deck.csv for the Latias ex / Xerneas ex Psychic-weakness-exploit deck.

Archetype rationale (see report): all 4 Pokemon lines are mono-Psychic Basics
(zero evolution setup), so every copy drawn is an immediately-playable
attacker. Latias ex's Skyliner ability gives every Basic in play free
retreat, letting us pivot fragile ex attackers out of danger without losing
tempo. Psychic attacks get the engine's automatic weakness-doubling against
Mega Lucario ex (Weakness {P}), the most common opponent in our rating band.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from card_data import CardTable
from deck_validate import validate_deck

Latias_ex = 184
Xerneas_ex = 331
Scream_Tail_ex = 969
Cresselia = 764

Ultra_Ball = 1121
Dusk_Ball = 1102
Lillie_Determination = 1227
Carmine = 1192
Crispin = 1198
Boss_Orders = 1182
Night_Stretcher = 1097
Switch = 1123
Waitress = 1235
Mystery_Garden = 1263
Poke_Pad = 1152

Basic_Psychic_Energy = 5

DECK = (
    [Latias_ex] * 4 + [Xerneas_ex] * 4 + [Scream_Tail_ex] * 4 + [Cresselia] * 4  # 16 Pokemon
    + [Ultra_Ball] * 4 + [Dusk_Ball] * 3 + [Lillie_Determination] * 4 + [Carmine] * 3
    + [Crispin] * 4 + [Boss_Orders] * 3 + [Night_Stretcher] * 2 + [Switch] * 2
    + [Waitress] * 3 + [Mystery_Garden] * 1  # 29 Trainers (dropped Poke Pad - only
    # fetches Cresselia given our 3 other lines are Rule-Box ex, so low value here;
    # traded for 1 extra Waitress and 2 more Energy to speed up reaching the 3-energy
    # threshold both main attack lines need)
    + [Basic_Psychic_Energy] * 15  # 15 Energy (up from 13)
)

if __name__ == "__main__":
    assert len(DECK) == 60, f"deck has {len(DECK)} cards"
    table = CardTable.load()
    result = validate_deck(DECK, table)
    print(f"validation: errorType={result.error_type} ({result.message})")
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "deck.csv")
    with open(out_path, "w") as f:
        f.write("\n".join(str(c) for c in DECK) + "\n")
    print(f"wrote {out_path} ({len(DECK)} cards)")
