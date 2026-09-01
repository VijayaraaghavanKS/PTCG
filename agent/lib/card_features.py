"""Feature engineering over EN Card Data.csv: parsed cost/damage, per-card
aggregation, and a generalized card-value score.

Adapted from the parsing/scoring approach in a public reference notebook for
this competition ("PTCG Strategy MRI: From 2,022 Cards to 60 Decisions",
clean-room implementation, no external deps beyond numpy/pandas/scipy) and
re-verified here against our own copy of the CSV (column names below match
what's actually in `pokemon-tcg-ai-battle/EN Card Data.csv`, including the
"Previos stage" typo present in the real file).
"""
from __future__ import annotations

import math
import re
from typing import Iterable

import numpy as np
import pandas as pd

from card_data import DEFAULT_CSV_PATH, TYPE_COL, load_moves


def parse_energy_cost(value: object) -> dict:
    text = "" if pd.isna(value) else str(value).strip()
    tokens = re.findall(r"\{[^{}]+\}|●", text)
    typed = tuple(t for t in tokens if t != "●")
    return {
        "energy_total": len(tokens),
        "energy_typed": len(typed),
        "energy_colorless": tokens.count("●"),
    }


def parse_damage(value: object) -> dict:
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return {"damage_value": np.nan, "damage_kind": "missing"}
    if re.fullmatch(r"\d+", text):
        return {"damage_value": float(int(text)), "damage_kind": "deterministic"}
    kind = "multiplier" if ("×" in text or "x" in text.lower()) else (
        "conditional_addition" if "+" in text else
        "modifier_or_reduction" if "-" in text else "unparsed"
    )
    return {"damage_value": np.nan, "damage_kind": kind}


def classify_card(stage_or_type: object) -> str:
    text = "" if pd.isna(stage_or_type) else str(stage_or_type).strip()
    if text in {"Item", "Supporter", "Pokémon Tool", "Stadium"}:
        return "Trainer"
    if text in {"Basic Pokémon", "Stage 1 Pokémon", "Stage 2 Pokémon"}:
        return "Pokemon"
    if "Energy" in text:
        return "Energy"
    return "Other"


def _first_nonempty(values: Iterable[object]) -> object:
    for v in values:
        if v is not None and not pd.isna(v) and str(v).strip():
            return v
    return np.nan


def _clean(value: object) -> str:
    return "" if pd.isna(value) else str(value).strip()


_KEYWORD_GROUPS = {
    "draw": ("draw",),
    "search": ("search", "look at"),
    "switch": ("switch", "retreat"),
    "energy": ("energy", "attach"),
    "heal": ("heal", "recover"),
    "disrupt": ("discard", "shuffle", "opponent"),
}


def _effect_keywords(text: str) -> dict:
    lowered = text.lower()
    return {name: int(any(tok in lowered for tok in toks)) for name, toks in _KEYWORD_GROUPS.items()}


def build_card_table(csv_path: str = DEFAULT_CSV_PATH) -> pd.DataFrame:
    """One row per Card ID with parsed cost/damage/effect features."""
    moves = load_moves(csv_path)
    cost = moves["Cost"].map(parse_energy_cost).apply(pd.Series)
    damage = moves["Damage"].map(parse_damage).apply(pd.Series)
    working = pd.concat([moves, cost, damage], axis=1)

    records = []
    for card_id, group in working.groupby("Card ID", sort=True):
        deterministic = group[group["damage_kind"] == "deterministic"]
        positive_cost = deterministic[deterministic["energy_total"] > 0]
        best_dpe = float((positive_cost["damage_value"] / positive_cost["energy_total"]).max()) if len(positive_cost) else np.nan

        effect_parts = []
        for v in group["Effect Explanation"]:
            t = _clean(v)
            if t and t not in effect_parts:
                effect_parts.append(t)
        effect_text = " | ".join(effect_parts)

        stage = _clean(_first_nonempty(group[TYPE_COL]))
        damage_nonmissing = group["damage_kind"] != "missing"
        record = {
            "Card ID": int(card_id),
            "Card Name": _clean(_first_nonempty(group["Card Name"])),
            "StageOrTrainerType": stage,
            "Rule": _clean(_first_nonempty(group["Rule"])),
            "PreviousStage": _clean(_first_nonempty(group["Previos stage"])),
            "HP": float(pd.to_numeric(group["HP"], errors="coerce").max()),
            "Type": _clean(_first_nonempty(group["Type"])),
            "Weakness": _clean(_first_nonempty(group["Weakness"])),
            "Resistance": _clean(_first_nonempty(group["Resistance (Type)"])),
            "Retreat": float(pd.to_numeric(group["Retreat"], errors="coerce").max()),
            "Kind": classify_card(stage),
            "MoveRows": int(group["Move Name"].notna().sum()),
            "ConditionalMoveRows": int((damage_nonmissing & (group["damage_kind"] != "deterministic")).sum()),
            "MaxDeterministicDamage": float(deterministic["damage_value"].max()) if len(deterministic) else np.nan,
            "BestDamagePerEnergy": best_dpe,
            "EffectText": effect_text,
            "EffectLength": len(effect_text),
        }
        record.update(_effect_keywords(effect_text))
        records.append(record)

    cards = pd.DataFrame.from_records(records).sort_values("Card ID").reset_index(drop=True)
    cards["IsBasicPokemon"] = cards["StageOrTrainerType"].eq("Basic Pokémon")
    cards["IsBasicEnergy"] = cards["StageOrTrainerType"].eq("Basic Energy")
    cards["IsAceSpec"] = cards["Rule"].eq("ACE SPEC")
    cards["EvolutionDepth"] = cards["StageOrTrainerType"].map(
        {"Basic Pokémon": 0, "Stage 1 Pokémon": 1, "Stage 2 Pokémon": 2}
    )
    return cards.set_index("Card ID", drop=False)


def robust_z(values: pd.Series, *, fill: float = 0.0) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = float(numeric.median()) if numeric.notna().any() else 0.0
    mad = float((numeric - median).abs().median()) if numeric.notna().any() else 0.0
    scale = 1.4826 * mad
    if not math.isfinite(scale) or scale < 1e-12:
        scale = float(numeric.std(ddof=0)) if numeric.notna().any() else 1.0
    if not math.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    return ((numeric.fillna(median) - median) / scale).clip(-3.0, 3.0).fillna(fill)


def score_cards(cards: pd.DataFrame) -> pd.DataFrame:
    """Generalized, config-driven card-value score (replaces per-card-ID hardcoding)."""
    scored = cards.copy()
    scored["durability_z"] = robust_z(scored["HP"])
    scored["pressure_z"] = robust_z(scored["BestDamagePerEnergy"])
    scored["mobility_z"] = -robust_z(scored["Retreat"])
    move_denom = scored["MoveRows"].replace(0, np.nan)
    scored["conditional_share"] = (scored["ConditionalMoveRows"] / move_denom).fillna(0.0)
    scored["trainer_signal"] = (
        1.00 * scored["draw"] + 1.15 * scored["search"] + 0.65 * scored["switch"]
        + 0.75 * scored["energy"] + 0.35 * scored["heal"] + 0.45 * scored["disrupt"]
    )
    pokemon_utility = (
        0.34 * scored["durability_z"] + 0.42 * scored["pressure_z"] + 0.18 * scored["mobility_z"]
        - 0.20 * scored["conditional_share"] - 0.10 * scored["EvolutionDepth"].fillna(0.0)
    )
    trainer_utility = robust_z(scored["trainer_signal"]) - 0.05 * robust_z(scored["EffectLength"])
    energy_utility = pd.Series(0.0, index=scored.index)
    scored["Utility"] = np.select(
        [scored["Kind"].eq("Pokemon"), scored["Kind"].eq("Trainer"), scored["Kind"].eq("Energy")],
        [pokemon_utility, trainer_utility, energy_utility],
        default=-3.0,
    )
    return scored


if __name__ == "__main__":
    cards = build_card_table()
    scored = score_cards(cards)
    print(f"{len(scored)} unique cards scored")
    print("\nTop 15 Pokemon by Utility:")
    print(scored[scored["Kind"] == "Pokemon"].nlargest(15, "Utility")[
        ["Card Name", "HP", "BestDamagePerEnergy", "Retreat", "Utility"]
    ].to_string(index=False))
    print("\nTop 10 Trainers by Utility:")
    print(scored[scored["Kind"] == "Trainer"].nlargest(10, "Utility")[
        ["Card Name", "StageOrTrainerType", "Utility"]
    ].to_string(index=False))
