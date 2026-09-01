"""
Generic, deck-agnostic feature extraction for scoring a candidate Option
given an Observation. Used both offline (training, on replay data) and
at inference time inside the ml_v1 agent, so the two MUST stay in sync.

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

FEATURE_DIM = N_SELECT_TYPE + N_SELECT_CONTEXT + N_OPTION_TYPE + N_AREA + N_CARDTYPE + N_NUMERIC


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

    assert len(out) == FEATURE_DIM, f"expected {FEATURE_DIM} total feats, got {len(out)}"
    return out
