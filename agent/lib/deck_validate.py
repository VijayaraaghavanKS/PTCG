"""Pure-Python mirror of the native engine's ApiBattleStart deck legality check.

Lets us reject an illegal deck instantly (no native engine round-trip) while
iterating on a deck-builder. Error codes match the real engine exactly
(confirmed against ptcg_engine/ptcgProgram 22/Api.h via Kaggle discussion
"ApiBattleStart rejects decks with a bare integer - here is what each code
means"):
  0 = valid
  1 = a card ID is not in the card table
  2 = more than 4 copies of one card NAME (grouped by name, not ID; only
      Basic Energy is exempt -- Special Energy IS capped like anything else)
  3 = no Basic Pokemon in the deck
  4 = more than one ACE SPEC card
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from card_data import CardTable, BASIC_ENERGY_TYPE, TYPE_COL


@dataclass
class DeckValidationResult:
    error_type: int  # 0-4, matches engine errorType
    message: str

    @property
    def ok(self) -> bool:
        return self.error_type == 0


def validate_deck(deck: list[int], table: CardTable, deck_size: int = 60) -> DeckValidationResult:
    if len(deck) != deck_size:
        return DeckValidationResult(1, f"deck has {len(deck)} cards, expected {deck_size}")

    for card_id in deck:
        if card_id not in table.id_to_name:
            return DeckValidationResult(1, f"card ID {card_id} is not in the card table")

    name_counts = Counter(table.name_of(cid) for cid in deck)
    for name, count in name_counts.items():
        rows = table.cards[table.cards["Card Name"] == name]
        is_basic_energy = bool(rows.iloc[0][TYPE_COL] == BASIC_ENERGY_TYPE) if len(rows) else False
        if not is_basic_energy and count > 4:
            return DeckValidationResult(2, f"{count} copies of '{name}' (limit 4, Basic Energy exempt)")

    if not any(table.cards.loc[cid, "is_basic_pokemon"] for cid in deck):
        return DeckValidationResult(3, "no Basic Pokemon in the deck")

    ace_spec_count = sum(1 for cid in deck if table.cards.loc[cid, "is_ace_spec"])
    if ace_spec_count > 1:
        return DeckValidationResult(4, f"{ace_spec_count} ACE SPEC cards (limit 1)")

    return DeckValidationResult(0, "valid")


if __name__ == "__main__":
    import sys

    table = CardTable.load()
    for deck_csv in sys.argv[1:]:
        with open(deck_csv) as f:
            deck = [int(line.strip()) for line in f if line.strip()]
        result = validate_deck(deck, table)
        print(f"{deck_csv}: errorType={result.error_type} ({result.message})")
