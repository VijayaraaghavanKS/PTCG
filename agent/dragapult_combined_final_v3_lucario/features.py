"""
Generic, deck-agnostic feature extraction for scoring a candidate Option
given an Observation. Used both offline (training, on replay data) and
at inference time inside the bc_v2 / bc_reranker_v2 agents, so the two
MUST stay in sync.

v2 change vs ml_v1/bc_v1's features.py: adds an explicit lethal/KO-aware
feature block (N_LETHAL below). The bc_v1 postmortem's stated hypothesis
for underperformance was "our feature set has no explicit lethal/KO-
sequencing signal the way the hand-tuned heuristic does" -- these features
make that signal directly visible to the model instead of forcing it to
approximate weakness/resistance-aware damage math implicitly from generic
stats. Damage/weakness math mirrors the generalized effective_damage()
pattern verified working in agent/latias_xerneas_v1/main.py (compares the
ATTACKING Pokemon's own type against the TARGET's Weakness/Resistance --
the correct TCG mechanic, not a hardcoded special case) and the KO/prize
bookkeeping mirrors agent/dragapult_day1/main.py's prize_count().

No third-party dependencies (stdlib only) so it can run inside the
Kaggle submission sandbox unmodified.
"""
import math

from cg.api import (
    AreaType, SelectType, SelectContext, OptionType, CardType, EnergyType,
    Observation, Option, Card, Pokemon,
)

N_SELECT_TYPE = 11
N_SELECT_CONTEXT = 49
N_OPTION_TYPE = 17
N_AREA = 12
N_CARDTYPE = 7

N_NUMERIC = 34
N_LETHAL = 8

FEATURE_DIM = N_SELECT_TYPE + N_SELECT_CONTEXT + N_OPTION_TYPE + N_AREA + N_CARDTYPE + N_NUMERIC + N_LETHAL


def _onehot(idx0based, n, out):
    v = [0.0] * n
    if idx0based is not None and 0 <= idx0based < n:
        v[idx0based] = 1.0
    out.extend(v)


def get_card(obs: Observation, area, index, player_index):
    """Safely fetch a Card/Pokemon object from a given zone. Returns None if not resolvable."""
    if area is None or index is None or player_index is None:
        return None
    try:
        ps = obs.current.players[player_index]
    except (IndexError, TypeError):
        return None
    try:
        if area == AreaType.DECK:
            return obs.select.deck[index] if obs.select.deck else None
        if area == AreaType.HAND:
            return ps.hand[index] if ps.hand else None
        if area == AreaType.DISCARD:
            return ps.discard[index]
        if area == AreaType.ACTIVE:
            return ps.active[index] if index < len(ps.active) else None
        if area == AreaType.BENCH:
            return ps.bench[index]
        if area == AreaType.PRIZE:
            return ps.prize[index]
        if area == AreaType.STADIUM:
            return obs.current.stadium[index]
        if area == AreaType.LOOKING:
            return obs.current.looking[index] if obs.current.looking else None
    except (IndexError, TypeError):
        return None
    return None


def _resolve_primary_card(obs, opt, my_index):
    """Best-effort resolution of the 'main' card referenced by an option, plus whose it is."""
    area = opt.area
    index = opt.index
    pidx = opt.playerIndex if opt.playerIndex is not None else my_index

    if area is None and opt.type == OptionType.PLAY:
        area = AreaType.HAND
        pidx = my_index
    if area is None and opt.type == OptionType.DISCARD:
        # index/area should be set per spec; fall back to hand
        area = AreaType.HAND
        pidx = my_index

    card = get_card(obs, area, index, pidx)
    return card, area, pidx


def _resolve_target_pokemon(obs, opt, my_index):
    """For ATTACH/EVOLVE style options: the in-play Pokemon being targeted."""
    if opt.inPlayArea is None or opt.inPlayIndex is None:
        return None
    return get_card(obs, opt.inPlayArea, opt.inPlayIndex, my_index)


def _prize_value(card_data):
    """How many prize cards a KO on this card would yield (engine rule: 3 for
    Mega ex, 2 for ex, 1 otherwise). Doesn't model per-card modifiers (e.g.
    Legacy Energy / Lillie's Pearl reductions) -- those are rare, deck-
    specific text effects a deck-agnostic extractor can't enumerate; this is
    a reference approximation, same spirit as dragapult_day1's prize_count()
    but without the hardcoded card-id modifiers that only apply to its own
    deck's own cards."""
    if card_data is None:
        return 1
    if getattr(card_data, "megaEx", False):
        return 3
    if getattr(card_data, "ex", False):
        return 2
    return 1


def _effective_damage(attacker_data, atk, target_data):
    """Weakness/resistance-aware damage estimate: compares the ATTACKING
    Pokemon's own type against the TARGET's Weakness/Resistance (weakness is
    a property of the attacker's type, not the energy that paid for the
    attack -- the correct TCG mechanic). Generalized, not hardcoded to any
    one type, so it's correct for whatever deck this extractor is run on."""
    if attacker_data is None or atk is None or target_data is None:
        return atk.damage if atk is not None else 0.0
    dmg = float(atk.damage or 0)
    if dmg <= 0:
        return dmg
    if target_data.weakness is not None and target_data.weakness == attacker_data.energyType:
        dmg *= 2
    elif target_data.resistance is not None and target_data.resistance == attacker_data.energyType:
        dmg = max(0.0, dmg - 30)
    return dmg


def _best_opponent_threat(op_active, op_active_data, target_data, card_table, attack_table):
    """Rough estimate of the most damage `op_active` could deal to a
    Pokemon with card-data `target_data` next turn (assumes one more energy
    attach), applying weakness doubling using the OPPONENT's own type
    against the TARGET's weakness -- same generalized mechanic as
    _effective_damage, just evaluated from the other side. Mirrors
    agent/latias_xerneas_v1/main.py's opponent_best_damage_estimate()."""
    if op_active is None or op_active_data is None:
        return 0.0
    have = len(op_active.energies)
    best = 0.0
    for aid in op_active_data.attacks:
        atk = attack_table.get(aid)
        if atk is None:
            continue
        req = len(atk.energies)
        if have + 1 >= req:  # assume one more energy attached next turn
            best = max(best, _effective_damage(op_active_data, atk, target_data))
    return best


def extract_features(obs: Observation, opt: Option, card_table: dict, attack_table: dict) -> list:
    """Build a fixed-length numeric feature vector for one candidate Option."""
    state = obs.current
    select = obs.select
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    out = []

    # --- one-hot: select.type, select.context, option.type ---
    _onehot(int(select.type) - 1 if select.type is not None else None, N_SELECT_TYPE, out)
    ctx = int(select.context) if select.context is not None else None
    # SelectContext values run 0..48 already (0-based)
    _onehot(ctx, N_SELECT_CONTEXT, out)
    otype = int(opt.type) if opt.type is not None else None
    _onehot(otype, N_OPTION_TYPE, out)

    # --- resolve referenced card(s) ---
    primary, p_area, p_pidx = _resolve_primary_card(obs, opt, my_index)
    target_pokemon = _resolve_target_pokemon(obs, opt, my_index)
    # Prefer target pokemon's card-data for ATTACH/EVOLVE "which pokemon benefits" framing,
    # but the card being played/attached is `primary`.
    ref_card_data = None
    if primary is not None:
        ref_card_data = card_table.get(primary.id)

    area_idx = (int(p_area) - 1) if p_area is not None else None
    _onehot(area_idx, N_AREA, out)

    cardtype_idx = (int(ref_card_data.cardType)) if ref_card_data is not None else None
    _onehot(cardtype_idx, N_CARDTYPE, out)

    # --- numeric features ---
    num = []

    def prize_remaining(ps):
        return len(ps.prize)

    num.append(state.turn / 30.0)
    num.append(prize_remaining(my_state) / 6.0)
    num.append(prize_remaining(op_state) / 6.0)
    num.append(my_state.handCount / 10.0)
    num.append(op_state.handCount / 10.0)
    num.append(len(my_state.bench) / 5.0)
    num.append(len(op_state.bench) / 5.0)

    def hp_ratio(ps):
        if ps.active and ps.active[0] is not None and ps.active[0].maxHp:
            return ps.active[0].hp / ps.active[0].maxHp
        return 0.0

    num.append(hp_ratio(my_state))
    num.append(hp_ratio(op_state))
    num.append(1.0 if state.supporterPlayed else 0.0)
    num.append(1.0 if state.stadiumPlayed else 0.0)
    num.append(1.0 if state.energyAttached else 0.0)
    num.append(1.0 if state.retreated else 0.0)
    num.append((select.minCount or 0) / 5.0)
    num.append((select.maxCount or 0) / 5.0)
    num.append((select.remainEnergyCost or 0) / 5.0)
    num.append((select.remainDamageCounter or 0) / 10.0)
    num.append((opt.number or 0) / 10.0 if opt.type == OptionType.NUMBER else 0.0)

    is_own = 1.0 if p_pidx == my_index else 0.0
    is_opp = 1.0 if p_pidx == (1 - my_index) else 0.0
    num.append(is_own)
    num.append(is_opp)

    # Pokemon-instance-level (only populated if `primary` is an in-play Pokemon)
    pkm = primary if isinstance(primary, Pokemon) else (target_pokemon if isinstance(target_pokemon, Pokemon) else None)
    if pkm is not None and pkm.maxHp:
        num.append(pkm.hp / pkm.maxHp)
        num.append(len(pkm.energies) / 6.0)
        num.append(len(pkm.tools) / 3.0)
        num.append(1.0 if pkm.appearThisTurn else 0.0)
    else:
        num.extend([0.0, 0.0, 0.0, 0.0])

    if ref_card_data is not None:
        num.append((ref_card_data.retreatCost or 0) / 4.0)
        num.append(1.0 if ref_card_data.basic else 0.0)
        num.append(1.0 if ref_card_data.stage1 else 0.0)
        num.append(1.0 if ref_card_data.stage2 else 0.0)
        num.append(1.0 if (ref_card_data.ex or ref_card_data.megaEx) else 0.0)
        prize_val = 3 if ref_card_data.megaEx else (2 if ref_card_data.ex else 1)
        num.append(prize_val / 3.0)
        num.append((ref_card_data.hp or 0) / 300.0)
    else:
        num.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    if opt.type == OptionType.ATTACK and opt.attackId is not None:
        atk = attack_table.get(opt.attackId)
        if atk is not None:
            num.append((atk.damage or 0) / 300.0)
            num.append(len(atk.energies) / 5.0)
        else:
            num.extend([0.0, 0.0])
    else:
        num.extend([0.0, 0.0])

    num.append(1.0)  # bias

    assert len(num) == N_NUMERIC, f"expected {N_NUMERIC} numeric feats, got {len(num)}"
    out.extend(num)

    # --- lethal / KO-aware block (v2 addition) ---
    lethal = _extract_lethal_features(obs, opt, card_table, attack_table, my_index, my_state, op_state, primary)
    assert len(lethal) == N_LETHAL, f"expected {N_LETHAL} lethal feats, got {len(lethal)}"
    out.extend(lethal)

    assert len(out) == FEATURE_DIM, f"expected {FEATURE_DIM} total feats, got {len(out)}"
    return out


def _extract_lethal_features(obs, opt, card_table, attack_table, my_index, my_state, op_state, primary):
    """8 features making lethal/near-lethal opportunities and KO danger
    directly visible to the model, computed generically (deck-agnostic --
    only uses card_table/attack_table lookups, never hardcoded card ids):

      0: atk_eff_damage_ratio   -- this option's implied attack's effective
                                    (weakness/resistance-aware) damage vs the
                                    opponent's active, /300. 0 for non-ATTACK
                                    options.
      1: atk_is_ko              -- 1.0 if that damage >= opponent active's
                                    current HP (this option KOs the active).
      2: atk_op_hp_after_ratio  -- opponent active's HP remaining after this
                                    attack, as a fraction of its max HP (0 if
                                    KO'd, ~1 if the attack barely scratches it)
                                    -- "how close to KO" signal even on non-
                                    lethal attacks.
      3: atk_prize_value_ratio  -- prize cards this KO would yield / 3 (0 if
                                    not a KO).
      4: atk_wins_game          -- 1.0 if this KO's prize count would meet or
                                    exceed our own remaining prizes (i.e. this
                                    attack could end the game right now).
      5: threat_ratio           -- opponent's best next-turn effective damage
                                    against OUR current active (assuming one
                                    more energy attach), /300. Same value for
                                    every option in a decision -- a general
                                    "how much danger are we in" signal.
      6: my_active_in_ko_range  -- 1.0 if that threat >= our active's current
                                    HP (we could be KO'd next opponent turn if
                                    we don't act).
      7: switch_target_ko_range -- for CARD options that pick our own new
                                    active Pokemon (SWITCH/TO_ACTIVE/
                                    SETUP_ACTIVE_POKEMON contexts): 1.0 if the
                                    candidate Pokemon would already be within
                                    the opponent's next-attack KO range. 0 for
                                    all other option types (no such target).
    """
    select = obs.select
    my_active = my_state.active[0] if my_state.active else None
    op_active = op_state.active[0] if (op_state.active and op_state.active[0] is not None) else None
    my_active_data = card_table.get(my_active.id) if my_active is not None else None
    op_active_data = card_table.get(op_active.id) if op_active is not None else None

    f0 = f1 = f2 = f3 = f4 = 0.0
    if opt.type == OptionType.ATTACK and opt.attackId is not None and my_active is not None and op_active is not None:
        atk = attack_table.get(opt.attackId)
        attacker_data = my_active_data
        if atk is not None and attacker_data is not None and op_active_data is not None:
            dmg = _effective_damage(attacker_data, atk, op_active_data)
            f0 = min(dmg, 600.0) / 300.0
            if op_active.hp is not None and op_active.hp > 0:
                is_ko = dmg >= op_active.hp
                f1 = 1.0 if is_ko else 0.0
                if op_active.maxHp:
                    f2 = max(0.0, op_active.hp - dmg) / op_active.maxHp
                if is_ko:
                    pv = _prize_value(op_active_data)
                    f3 = pv / 3.0
                    if len(my_state.prize) <= pv:
                        f4 = 1.0

    f5 = f6 = 0.0
    if op_active is not None and op_active_data is not None and my_active is not None and my_active_data is not None:
        threat = _best_opponent_threat(op_active, op_active_data, my_active_data, card_table, attack_table)
        f5 = min(threat, 600.0) / 300.0
        if my_active.hp is not None and my_active.hp > 0 and threat >= my_active.hp:
            f6 = 1.0

    f7 = 0.0
    if (opt.type == OptionType.CARD and select is not None
            and select.context in (SelectContext.SWITCH, SelectContext.TO_ACTIVE, SelectContext.SETUP_ACTIVE_POKEMON)
            and (opt.playerIndex is None or opt.playerIndex == my_index)):
        candidate = get_card(obs, opt.area, opt.index, my_index)
        if isinstance(candidate, Pokemon) and op_active is not None and op_active_data is not None:
            candidate_data = card_table.get(candidate.id)
            if candidate_data is not None:
                threat = _best_opponent_threat(op_active, op_active_data, candidate_data, card_table, attack_table)
                if candidate.hp is not None and candidate.hp > 0 and threat >= candidate.hp:
                    f7 = 1.0

    return [f0, f1, f2, f3, f4, f5, f6, f7]
