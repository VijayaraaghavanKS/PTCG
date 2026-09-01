"""Load and index EN Card Data.csv.

Gotchas learned from Kaggle discussion (thread: "ApiBattleStart rejects decks
with a bare integer"), verified against the shipped CSV and Api.h:
  - The file starts with a UTF-8 BOM; pandas needs encoding="utf-8-sig" or the
    first column reads as " Card ID" (leading space) instead of "Card ID".
  - The card-type column is "Stage (Pokémon)/Type (Energy and Trainer)", not
    "Category" (Category is NaN for ~80% of rows and encodes flavor subtypes
    like "Tera(Fire)" or "Trainer's Pokémon(N)", not the engine's type check).
  - One row per *move* (attack/ability), not per card: 2022 rows, 1267 unique
    Card IDs. Multi-attack Pokemon have 2 rows sharing the same Card ID.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

DEFAULT_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..",
    "pokemon-tcg-ai-battle", "EN Card Data.csv",
)

TYPE_COL = "Stage (Pokémon)/Type (Energy and Trainer)"

POKEMON_STAGES = {"Basic Pokémon", "Stage 1 Pokémon", "Stage 2 Pokémon"}
BASIC_ENERGY_TYPE = "Basic Energy"
SPECIAL_ENERGY_TYPE = "Special Energy"


def load_moves(csv_path: str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """One row per move/attack/ability, exactly as shipped."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df["is_basic_pokemon"] = df[TYPE_COL] == "Basic Pokémon"
    df["is_pokemon"] = df[TYPE_COL].isin(POKEMON_STAGES)
    df["is_ace_spec"] = df["Rule"] == "ACE SPEC"
    df["is_basic_energy"] = df[TYPE_COL] == BASIC_ENERGY_TYPE
    return df


def load_cards(csv_path: str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """One row per unique Card ID (collapses multi-move Pokemon).

    Non-move fields (Name/Expansion/Stage/HP/Type/Weakness/Resistance/Retreat/
    Rule/Category) are identical across a card's move-rows, so `first()` is
    safe. `moves` collects the per-attack (Move Name, Cost, Damage) tuples.
    """
    moves = load_moves(csv_path)
    static_cols = [
        "Card ID", "Card Name", "Expansion", TYPE_COL, "Rule", "Category",
        "Previos stage", "HP", "Type", "Weakness", "Resistance (Type)",
        "Retreat", "is_basic_pokemon", "is_pokemon", "is_ace_spec", "is_basic_energy",
    ]
    cards = moves.groupby("Card ID", as_index=False)[static_cols[1:]].first()
    move_lists = (
        moves.groupby("Card ID")[["Move Name", "Cost", "Damage", "Effect Explanation"]]
        .apply(lambda g: g.to_dict("records"))
        .rename("moves")
    )
    cards = cards.merge(move_lists, on="Card ID")
    return cards.set_index("Card ID", drop=False)


@dataclass
class CardTable:
    cards: pd.DataFrame  # indexed by Card ID, one row per card
    id_to_name: dict[int, str]
    name_to_ids: dict[str, list[int]]  # for the per-NAME copy-limit check

    @classmethod
    def load(cls, csv_path: str = DEFAULT_CSV_PATH) -> "CardTable":
        cards = load_cards(csv_path)
        id_to_name = cards["Card Name"].to_dict()
        name_to_ids: dict[str, list[int]] = {}
        for cid, name in id_to_name.items():
            name_to_ids.setdefault(name, []).append(cid)
        return cls(cards=cards, id_to_name=id_to_name, name_to_ids=name_to_ids)

    def name_of(self, card_id: int) -> str | None:
        return self.id_to_name.get(card_id)
