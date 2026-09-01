import os
import sys
import time
import random
from collections import defaultdict

from cg.api import AreaType, CardType, Log, LogType, Observation, SelectContext, OptionType, Card, Pokemon, State, all_card_data, all_attack, to_observation_class

# Search API is used ONLY for a turn-local "plan-then-subordinate" search (see
# compute_turn_plan below) - never for adversarial/opponent-reply lookahead.
# Hard-gated: any import/runtime failure just disables planning and the agent
# falls back to the proven dragapult_day1 per-decision heuristic untouched.
try:
    from cg.api import search_begin, search_step, search_end
    _SEARCH_IMPORT_OK = True
except Exception:
    _SEARCH_IMPORT_OK = False
_search_available = True  # flips false permanently on first hard search failure

"""
Dragapult ex Deck (Plan v1)
Advanced Level
This deck focuses on setting up multiple knockouts to take at least three Prize cards in a single turn with its Phantom Dive attack.

PLAN v1 ADDITION (on top of the proven dragapult_day1 heuristic):
At the first MAIN-phase decision of each of our turns, run a small turn-local
search (via cg.api's search_begin/search_step, chained only through OUR OWN
remaining decisions this turn - never into the opponent's reply) to pick one
target "turn plan": which Pokemon ends up attacking, with which attack, and
which trainers/evolves/attaches the winning branch used to get there. That
plan is stored as persistent state (`turn_plan`) across the individual
agent(obs) calls that make up executing the turn. Every subsequent decision
this turn gets a small ADDITIVE score bonus if it progresses the plan (same
attacking Pokemon's evolve/attach line, the planned attack, or a trainer the
winning branch played) - it never overrides or replaces the base heuristic's
own scores/gates (which is what actually enforces legality/soundness), so if
the plan is wrong, stale, or covers nothing relevant to a given option, the
bonus is just 0 and behavior is IDENTICAL to plain dragapult_day1. The plan
is invalidated (falls back to pure base heuristic for the rest of the turn)
the moment its planned attacker is no longer found on our field.
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
    
# Load all card data from the API's helper function
all_card = all_card_data()
# Create a lookup table (dictionary) to quickly access card data by its cardId
card_table = {c.cardId:c for c in all_card}
# Attack lookup table (id -> Attack, incl. damage) - used only by the turn-plan
# rollout scorer below to prefer the strongest simultaneously-legal attack
# (e.g. Dragapult ex's Jet Headbutt is ALWAYS legal whenever Phantom Dive is,
# since it only needs 1 any-type energy - without a damage-aware tiebreak the
# rollout could lock a plan onto the weak attack).
try:
    attack_table = {a.attackId: a for a in all_attack()}
except Exception:
    attack_table = {}

# Decklist
Dreepy = 119  # ×4
Drakloak = 120  # ×4
Dragapult_ex = 121  # ×3
Fezandipiti_ex = 140  # ×1
Latias_ex = 184  # ×1
Budew = 235  # ×2
Meowth_ex = 1071  # ×1
Rare_Candy = 1079  # ×2
Unfair_Stamp = 1080  # ×1
Buddy_Buddy_Poffin = 1086  # ×4
Night_Stretcher = 1097  # ×2
Crushing_Hammer = 1120  # ×4
Ultra_Ball = 1121  # ×4
Poke_Pad = 1152  # x3
Lucky_Helmet = 1156  # ×1
Boss_Orders = 1182  # ×3
Crispin = 1198  # ×4
Brock_Scouting = 1210  # ×2
Lillie_Determination = 1227  # ×4
Team_Rocket_Watchtower = 1256  # ×2
Basic_Fire_Energy = 2  # ×4
Basic_Psychic_Energy = 5  # ×4

UNNECESSARY = -10000000

class AttackPlan:
    attack: int = 0
    counter: list[int] = []

can_switch = False
can_attack = False
can_main_attack = False
can_energy_attach = False
use_support = 0  # The Supporter card planned for use.
bench_attacker = False  # Whether there is a Benched Pokémon that is ready to attack
pre_turn_log: list[Log] = []
current_turn_log: list[Log] = []

prize: list[int] = []
card_counts: defaultdict[int, int] = defaultdict(int)
serial_set: set[int] = set()
plan_a = AttackPlan()
plan_b = AttackPlan()

# --- Turn plan state (plan-then-subordinate) ---
# None until first computed; otherwise a dict with keys:
#   turn, valid, score, attack_id, attacker_serial, attacker_id_final,
#   evolve_serials (list[int]), attach_serials (list[int]), trainer_seq (list[int])
# `valid=False` is a cached "already tried and found nothing usable this
# turn" sentinel so we don't re-run the search on every decision of the turn.
turn_plan: dict | None = None
_PLAN_DEBUG = os.environ.get("PLAN_DEBUG") == "1"
_PLAN_ABLATE = os.environ.get("PLAN_ABLATE", "")  # temporary ablation harness for local debugging only

PLAN_TIME_BUDGET_S = 0.35
PLAN_MAX_STEPS = 30
PLAN_MAX_CANDIDATES = 6
PLAN_FILLER_ENERGY = 2  # Basic_Fire_Energy id - always a legal, cheap filler card

# --- v1.1: gated attack-now bonus (see _plan_score_adjustment below) ---
# Unlike the flat/unconditional attack bonus tried and rejected in v1 (see
# AGENT_LOG - catastrophic 5/30 because it overrode the base heuristic's
# "develop board before attacking" priority), this bonus only ever fires in
# two narrow, independently-evidenced-safe circumstances: (1) the rollout's
# winning branch reaches an actual game win this turn (PLAN_BONUS_WIN), or
# (2) the winning branch scores a real KO this turn AND our current active
# is estimated to be in lethal danger from the opponent's own next attack
# (PLAN_BONUS_DANGER) - i.e. attacking now to bank value beats losing the
# attacker for nothing while "saving up". In every other case (no danger,
# no immediate KO) the bonus is 0, identical to the proven-safe v1 behavior.
PLAN_BONUS_WIN = 50000
PLAN_BONUS_DANGER = 900


def no_damage_dex(id: int) -> bool:
    """Checks if the defending Pokémon possesses innate immunities preventing Dragapult ex from hitting it."""
    # Drednaw, Milotic ex, Sylveon, Crustle
    return id == 158 or id == 207 or id == 330 or id == 345


def no_damage_counter(pokemon: Pokemon) -> bool:
    """Checks if a target prevents placement of Phantom Dive's 6 bench damage counters (via abilities/Energy)."""
    # Poltchageist, Empoleon ex, Skeledirge, Milotic ex, Misty's Magikarp, Antique Cover Fossil
    if pokemon.id == 28 or pokemon.id == 199 or pokemon.id == 203 or pokemon.id == 207 or pokemon.id == 362 or pokemon.id == 1136:
        return True
    for card in pokemon.energyCards:
        # Mist Energy, Rock Fighting Energy
        if card.id == 11 or card.id == 20:
            return True
    return False


def prize_count(pokemon: Pokemon, is_attack_damage: bool) -> int:
    """Calculates how many Prize cards a Pokémon yields upon being Knocked Out, factoring in modifiers."""
    data = card_table[pokemon.id]
    count = 3 if data.megaEx else 2 if data.ex else 1
    if is_attack_damage:
        for card in pokemon.energyCards:
            if card.id == 12:  # Legacy Energy
                count -= 1
        for card in pokemon.tools:
            if card.id == 1172 and "Lillie" in data.name:  # Lillie’s Pearl
                count -= 1
    return max(0, count)


def pokemon_score(pokemon: Pokemon, is_attack_damage: bool) -> int:
    """Heuristically evaluates the tactical worth of targeting a specific Pokémon on the opponent's field."""
    data = card_table[pokemon.id]
    score = prize_count(pokemon, is_attack_damage) * 1000
    score += len(pokemon.energies) * 150
    score += len(pokemon.tools) * 100
    if data.stage2:
        score += 250
    elif data.stage1:
        score += 130
    
    id = pokemon.id
    # Noctowl, Fan Rotom, Archaludon ex, Meowth ex
    if id == 173 or id == 174 or id == 190 or id == 1071:
        score -= 200
    if id == 112 and len(pokemon.energies) >= 1:  # Munkidori
        score += 300
    score += pokemon.hp
    return score


def add_card_count(card: Card | Pokemon | None, my_index: int):
    if card == None:
        return
    if isinstance(card, Pokemon) or card.playerIndex == my_index:
        if card.serial not in serial_set:
            card_counts[card.id] -= 1
            serial_set.add(card.serial)
    if isinstance(card, Pokemon):
        for c in card.energyCards:
            add_card_count(c, my_index)
        for c in card.tools:
            add_card_count(c, my_index)
        for c in card.preEvolution:
            add_card_count(c, my_index)

def set_card_counts(obs: Observation, my_index: int):
    card_counts.clear()
    serial_set.clear()
    for id in my_deck:
        card_counts[id] += 1
    
    state = obs.current
    my_state = state.players[my_index]
    for card in my_state.hand:
        add_card_count(card, my_index)
    for card in my_state.discard:
        add_card_count(card, my_index)
    for card in my_state.bench:
        add_card_count(card, my_index)
    for card in my_state.active:
        add_card_count(card, my_index)
    for card in state.stadium:
        add_card_count(card, my_index)
    if state.looking != None:
        for card in state.looking:
            add_card_count(card, my_index)
    add_card_count(obs.select.effect, my_index)

    
def get_card(obs: Observation, area: AreaType, index: int, player_index: int) -> Pokemon | Card | None:
    """Helper function to safely extract a Card or Pokemon object from specific zones."""
    ps = obs.current.players[player_index]
    match area:
        case AreaType.DECK:
            return obs.select.deck[index]
        case AreaType.HAND:
            return ps.hand[index]
        case AreaType.DISCARD:
            return ps.discard[index]
        case AreaType.ACTIVE:
            return ps.active[index]
        case AreaType.BENCH:
            return ps.bench[index]
        case AreaType.PRIZE:
            return ps.prize[index]
        case AreaType.STADIUM:
            return obs.current.stadium[index]
        case AreaType.LOOKING:
            return obs.current.looking[index]
        case _:
            return None


# =====================================================================
# Plan-then-subordinate: turn-local search + plan-serving score bonuses
# =====================================================================

def _unseen_own_pool(deck_counts: defaultdict) -> list[int]:
    """Flat list of our own not-yet-seen card ids (deck+facedown prize),
    per the same tracking the base heuristic already maintains."""
    pool = []
    for cid, cnt in deck_counts.items():
        if cnt > 0:
            pool.extend([cid] * cnt)
    if not pool:
        pool = [PLAN_FILLER_ENERGY]
    return pool


def _plan_generic_scores(obs: Observation, my_index: int) -> list[float]:
    """Coarse, cheap "make reasonable progress" scorer used ONLY to auto-pilot
    the rollout branches explored while searching for a turn plan. It does not
    need full domain fidelity - its only job is to find a plausible, good
    reachable end-of-turn state; the real decisions during actual play always
    go through the full proven heuristic (with a plan-serving bonus layered
    on top), never this function."""
    select = obs.select
    scores = []
    for o in select.option:
        s = 0.0
        try:
            if o.type == OptionType.ATTACK:
                dmg = 0
                atk = attack_table.get(o.attackId)
                if atk is not None:
                    dmg = atk.damage
                s = 2000000.0 + dmg  # prefer the strongest simultaneously-legal attack
            elif o.type == OptionType.EVOLVE:
                s = 6000.0
            elif o.type == OptionType.ATTACH:
                s = 4000.0
            elif o.type == OptionType.ABILITY:
                s = 3500.0
            elif o.type == OptionType.PLAY:
                card = get_card(obs, AreaType.HAND, o.index, my_index)
                cid = card.id if card is not None else 0
                if cid in (Rare_Candy, Ultra_Ball, Buddy_Buddy_Poffin, Poke_Pad):
                    s = 5000.0
                elif cid in (Basic_Fire_Energy, Basic_Psychic_Energy):
                    s = 4500.0
                else:
                    data = card_table.get(cid)
                    if data is not None and data.cardType == CardType.SUPPORTER:
                        s = 2500.0
                    else:
                        s = 1200.0
            elif o.type == OptionType.RETREAT:
                s = -2000.0
            elif o.type == OptionType.END:
                s = -5000.0
            elif o.type == OptionType.CARD:
                card = get_card(obs, o.area, o.index, o.playerIndex)
                if isinstance(card, Pokemon):
                    s = float(card.hp) + len(card.energies) * 40.0
                else:
                    s = 100.0
            elif o.type in (OptionType.ENERGY_CARD, OptionType.ENERGY):
                s = 20.0
            elif o.type == OptionType.YES:
                s = 1.0
            elif o.type == OptionType.NUMBER:
                s = float(o.number or 0)
            else:
                s = 0.0
        except Exception:
            s = 0.0
        scores.append(s)
    return scores


def _plan_pick(obs: Observation, my_index: int) -> list[int]:
    select = obs.select
    n = len(select.option)
    if n == 0:
        return []
    try:
        scores = _plan_generic_scores(obs, my_index)
    except Exception:
        scores = [0.0] * n
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
    k = min(select.maxCount, n)
    if k <= 0:
        k = min(1, n)
    return ranked[:k]


def _op_best_attack_damage(op_active: Pokemon | None, my_active: Pokemon | None) -> int:
    """Cheap, purely analytic (no search) estimate of the most damage the
    opponent's active Pokemon could plausibly deal to our active on their
    very next turn. Used ONLY to gate the plan's attack-now-vs-wait bonus
    below - self-contained (reuses only the already-loaded card_table /
    attack_table), no belief-tracking, no extra search calls, so it's cheap
    enough to call on every plan computation. Deliberately conservative: a
    forgiving +1 energy allowance (they could attach one more before
    attacking) so we don't under-estimate danger, and correct TCG weakness/
    resistance mechanics (a property of the ATTACKING Pokemon's own type,
    checked against our Pokemon's weakness/resistance - not the energy used
    to pay for the attack)."""
    if op_active is None or my_active is None:
        return 0
    op_data = card_table.get(op_active.id)
    my_data = card_table.get(my_active.id)
    if op_data is None or my_data is None:
        return 0
    have = len(op_active.energies)
    best = 0
    for atk_id in op_data.attacks:
        atk = attack_table.get(atk_id)
        if atk is None:
            continue
        cost = len(atk.energies)
        if have + 1 < cost:
            continue  # even attaching one more energy next turn wouldn't reach this attack
        dmg = atk.damage
        if my_data.weakness is not None and op_data.energyType == my_data.weakness:
            dmg *= 2
        elif my_data.resistance is not None and op_data.energyType == my_data.resistance:
            dmg = max(0, dmg - 30)
        best = max(best, dmg)
    return best


def _plan_attacker_in_danger(state: State, my_index: int, op_index: int) -> bool:
    """Is our CURRENT active Pokemon estimated to be within lethal range of
    the opponent's own next attack? Only the active can attack per TCG
    rules, so this is the only Pokemon relevant to gating an attack-now
    bonus. Fails safe (returns False, i.e. no danger bonus) on any missing
    data - the danger bonus is a nice-to-have, never a thing worth risking
    an exception over."""
    try:
        me = state.players[my_index]
        op = state.players[op_index]
        my_active = me.active[0] if me.active else None
        op_active = op.active[0] if op.active else None
        if my_active is None or op_active is None:
            return False
        return _op_best_attack_damage(op_active, my_active) >= my_active.hp
    except Exception:
        return False


def compute_turn_plan(obs: Observation, my_index: int, deck_counts: defaultdict) -> dict | None:
    """Fully solve (near-exhaustively, within a small time/step budget) for
    ONE fixed end-of-turn attack goal, by branching over this turn's current
    candidate actions and auto-piloting the rest of OUR OWN turn forward via
    the engine's real Search API - never stepping into the opponent's reply.
    Returns a plan dict (see `turn_plan` docstring above) or None on any
    failure/timeout/no-opportunity, in which case the caller falls back to
    the plain per-decision heuristic for the whole turn."""
    global _search_available
    if not (_SEARCH_IMPORT_OK and _search_available):
        return None
    if getattr(obs, "search_begin_input", None) is None:
        return None
    select = obs.select
    if select is None or select.context != SelectContext.MAIN or not select.option:
        return None

    state = obs.current
    op_index = 1 - my_index
    op_state = state.players[op_index]
    my_state = state.players[my_index]
    start_op_prize = len(op_state.prize)
    start_my_prize = len(my_state.prize)

    opts = select.option
    cand_idx = []
    for i, o in enumerate(opts):
        if o.type in (OptionType.ATTACK, OptionType.EVOLVE):
            cand_idx.append(i)
    for i, o in enumerate(opts):
        if len(cand_idx) >= PLAN_MAX_CANDIDATES:
            break
        if i in cand_idx:
            continue
        if o.type in (OptionType.ATTACH, OptionType.PLAY, OptionType.ABILITY):
            cand_idx.append(i)
    if not cand_idx:
        return None  # nothing plan-worthy this turn (pure forced/retreat/end options)
    cand_idx = cand_idx[:PLAN_MAX_CANDIDATES]

    try:
        pool = _unseen_own_pool(deck_counts)
        random.shuffle(pool)
        n_hidden_prize = sum(1 for c in my_state.prize if c is None)
        need = my_state.deckCount + n_hidden_prize
        if len(pool) < need:
            pool = pool + [PLAN_FILLER_ENERGY] * (need - len(pool))
        my_deck_guess = pool[:my_state.deckCount]
        fill_iter = iter(pool[my_state.deckCount:need])
        my_prize_guess = [c.id if c is not None else next(fill_iter, PLAN_FILLER_ENERGY) for c in my_state.prize]

        op_deck_guess = [PLAN_FILLER_ENERGY] * op_state.deckCount
        op_prize_guess = [PLAN_FILLER_ENERGY] * len(op_state.prize)
        op_hand_guess = [PLAN_FILLER_ENERGY] * op_state.handCount
        op_active_guess = []
        if op_state.active and op_state.active[0] is None:
            op_active_guess = [Dreepy]
    except Exception:
        return None

    t0 = time.monotonic()
    deadline = t0 + PLAN_TIME_BUDGET_S
    best = None

    try:
        ss0 = search_begin(obs, your_deck=my_deck_guess, your_prize=my_prize_guess,
                            opponent_deck=op_deck_guess, opponent_prize=op_prize_guess,
                            opponent_hand=op_hand_guess, opponent_active=op_active_guess)
    except Exception:
        _search_available = False
        return None
    root_sid = ss0.searchId

    try:
        for ci in cand_idx:
            if time.monotonic() > deadline:
                break
            first_opt = opts[ci]
            try:
                ss = search_step(root_sid, [ci])
            except Exception:
                continue
            sid, cur = ss.searchId, ss.observation

            attack_id_hit = None
            attacker_serial_hit = None
            attacker_id_final = None
            evolve_serials = []
            attach_serials = []
            trainer_seq = []

            if first_opt.type == OptionType.EVOLVE:
                pk = get_card(obs, first_opt.inPlayArea, first_opt.inPlayIndex, my_index)
                if isinstance(pk, Pokemon):
                    evolve_serials.append(pk.serial)
            elif first_opt.type == OptionType.ATTACH:
                pk = get_card(obs, first_opt.inPlayArea, first_opt.inPlayIndex, my_index)
                if isinstance(pk, Pokemon):
                    attach_serials.append(pk.serial)
            elif first_opt.type == OptionType.PLAY:
                card = get_card(obs, AreaType.HAND, first_opt.index, my_index)
                if card is not None:
                    trainer_seq.append(card.id)
            elif first_opt.type == OptionType.ATTACK:
                attack_id_hit = first_opt.attackId

            steps = 0
            while steps < PLAN_MAX_STEPS and time.monotonic() < deadline:
                cs = cur.current
                if cs is None or (cs.result is not None and cs.result >= 0):
                    break
                if cs.yourIndex != my_index or cur.select is None:
                    break  # turn passed on (or a forced opponent decision) - stop here
                csel = cur.select
                try:
                    picks = _plan_pick(cur, my_index)
                except Exception:
                    break
                if not picks:
                    break
                o = csel.option[picks[0]]

                if o.type == OptionType.ATTACK and attack_id_hit is None:
                    attack_id_hit = o.attackId
                    try:
                        for p in list(cs.players[my_index].active) + list(cs.players[my_index].bench):
                            if p is not None and o.attackId in card_table[p.id].attacks:
                                attacker_serial_hit = p.serial
                                attacker_id_final = p.id
                                break
                    except Exception:
                        pass
                elif o.type == OptionType.EVOLVE and attacker_serial_hit is None:
                    pk = get_card(cur, o.inPlayArea, o.inPlayIndex, my_index)
                    if isinstance(pk, Pokemon) and pk.serial not in evolve_serials:
                        evolve_serials.append(pk.serial)
                elif o.type == OptionType.ATTACH and attacker_serial_hit is None:
                    pk = get_card(cur, o.inPlayArea, o.inPlayIndex, my_index)
                    if isinstance(pk, Pokemon) and pk.serial not in attach_serials:
                        attach_serials.append(pk.serial)
                elif o.type == OptionType.PLAY and len(trainer_seq) < 6:
                    card = get_card(cur, AreaType.HAND, o.index, my_index)
                    if card is not None:
                        trainer_seq.append(card.id)

                is_end = (o.type == OptionType.END)
                try:
                    ss2 = search_step(sid, picks)
                except Exception:
                    break
                sid, cur = ss2.searchId, ss2.observation
                steps += 1
                if is_end:
                    break

            cs = cur.current
            score = -1e9
            wins_game = False
            this_turn_prizes_taken = 0
            if cs is not None:
                if cs.result is not None and cs.result >= 0:
                    score = 1e7 if cs.result == my_index else (0.0 if cs.result == 2 else -1e7)
                    wins_game = (cs.result == my_index)
                else:
                    me = cs.players[my_index]
                    op = cs.players[op_index]
                    my_field = [p for p in (list(me.active) + list(me.bench)) if p]
                    op_field = [p for p in (list(op.active) + list(op.bench)) if p]
                    my_hp = sum(p.hp for p in my_field)
                    op_hp = sum(p.hp for p in op_field)
                    my_en = sum(len(p.energies) for p in my_field)
                    no_active = 0 if (me.active and me.active[0] is not None) else 1
                    prizes_taken = max(0, start_op_prize - len(op.prize))
                    prizes_lost = max(0, start_my_prize - len(me.prize))
                    this_turn_prizes_taken = prizes_taken
                    score = (prizes_taken * 200000.0
                             - prizes_lost * 150000.0
                             + (my_hp - op_hp) * 1.0
                             + my_en * 25.0
                             - no_active * 300000.0)
                    if attack_id_hit is not None:
                        score += 500.0
                    if _PLAN_DEBUG:
                        print(f"PLAN_DEBUG   cand={ci} type={opts[ci].type} score={score} "
                              f"prizes_taken={prizes_taken} prizes_lost={prizes_lost} no_active={no_active} "
                              f"attack_id_hit={attack_id_hit} steps={steps}", file=sys.stderr)

            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "attack_id": attack_id_hit,
                    "attacker_serial": attacker_serial_hit,
                    "attacker_id_final": attacker_id_final,
                    "evolve_serials": evolve_serials,
                    "attach_serials": attach_serials,
                    "trainer_seq": trainer_seq,
                    "wins_game": wins_game,
                    "this_turn_prizes_taken": this_turn_prizes_taken,
                }
    except Exception as e:
        if _PLAN_DEBUG:
            import traceback
            print(f"PLAN_DEBUG compute_turn_plan EXCEPTION: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        best = None
    finally:
        try:
            search_end()
        except Exception:
            pass

    if best is None:
        if _PLAN_DEBUG:
            print(f"PLAN_DEBUG turn={state.turn} plan=None (no candidate found)", file=sys.stderr)
        return None
    best["turn"] = state.turn
    best["valid"] = True
    try:
        best["attacker_in_danger"] = _plan_attacker_in_danger(state, my_index, op_index)
    except Exception:
        best["attacker_in_danger"] = False
    if _PLAN_DEBUG:
        print(f"PLAN_DEBUG turn={state.turn} plan={best}", file=sys.stderr)
    return best


def _plan_feasible(plan: dict, obs: Observation, my_index: int) -> bool:
    """Cheap re-check run on every subsequent MAIN decision this turn: has
    the planned attacker unexpectedly disappeared (KO'd, forced away, etc.)?
    If so the plan degrades gracefully - it's dropped for the rest of the
    turn and every option falls back to a 0 bonus (pure base heuristic)."""
    if not plan.get("valid", False):
        return False
    attacker_serial = plan.get("attacker_serial")
    if attacker_serial is None:
        return True
    try:
        my_state = obs.current.players[my_index]
        for p in list(my_state.active) + list(my_state.bench):
            if p is not None and p.serial == attacker_serial:
                return True
    except Exception:
        return False
    return False


def _plan_score_adjustment(o, obs: Observation, context, my_index: int, my_state, plan: dict | None) -> int:
    """Small ADDITIVE bonus nudging THIS turn's remaining decisions toward
    the already-committed turn plan. Returns 0 whenever the plan is absent,
    stale, or doesn't cover this option - i.e. this can only ever bias which
    legal/positive-scored option the base heuristic prefers, never make it
    choose something the base heuristic already scored as illegal/negative
    (UNNECESSARY = -10,000,000 stays dominant over any bonus here)."""
    if plan is None or not plan.get("valid", False):
        return 0
    attacker_serial = plan.get("attacker_serial")
    ablate = _PLAN_ABLATE
    try:
        if o.type == OptionType.ATTACK:
            # v1 finding (see AGENT_LOG): an UNCONDITIONAL bonus here was
            # catastrophic (5/30 vs day1) because it overrode the base
            # heuristic's own type-priority ordering (setup actions score in
            # the thousands-to-tens-of-thousands range, ATTACK scores only
            # its small raw attackId) - the exact mechanism that correctly
            # sequences "develop board first, attack once nothing else is
            # more valuable this turn". That finding stands: this branch
            # still returns 0 (no bonus) in the default/no-danger/no-win
            # case, identical to the proven-safe v1 behavior.
            #
            # v1.1 addition: TWO narrowly-gated exceptions, each independently
            # ablation-tested (see AGENT_LOG). Neither is a flat preference
            # for attacking - both require the rollout to have already found
            # a concrete reason THIS SPECIFIC attack is worth taking now
            # rather than after more development:
            if plan.get("attack_id") is None or o.attackId != plan.get("attack_id"):
                return 0
            # (1) The rollout's winning branch reaches an outright game win
            # this turn. Taking a game-winning attack now (rather than risking
            # a same-turn detour into a lower-scored-but-positive setup action,
            # since the raw ATTACK option score is tiny) is unconditionally
            # correct - there is no plausible "wait" branch that beats winning
            # immediately.
            if "W" not in ablate and plan.get("wins_game", False):
                return PLAN_BONUS_WIN
            # (2) The rollout's winning branch scores a real KO this turn AND
            # our current active is estimated to be in lethal danger from the
            # opponent's own next attack (see _plan_attacker_in_danger) - i.e.
            # attacking now to bank real value beats losing this attacker for
            # nothing while "saving up" for a bigger turn that may never come.
            # Deliberately small (900) relative to setup-action scores
            # (1500-70000) - it can only break a close tie toward attacking
            # when genuinely under threat, never override a clearly-better
            # development play.
            if "D" not in ablate and plan.get("this_turn_prizes_taken", 0) > 0 and plan.get("attacker_in_danger", False):
                return PLAN_BONUS_DANGER
            return 0
        elif o.type == OptionType.EVOLVE:
            if "E" in ablate:
                return 0
            pk = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if isinstance(pk, Pokemon):
                if attacker_serial is not None and pk.serial == attacker_serial:
                    return 2500
                if pk.serial in plan.get("evolve_serials", []):
                    return 2500
        elif o.type == OptionType.ATTACH:
            if "A" in ablate:
                return 0
            pk = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            if isinstance(pk, Pokemon):
                if attacker_serial is not None and pk.serial == attacker_serial:
                    return 1500
                if pk.serial in plan.get("attach_serials", []):
                    return 1500
        elif o.type == OptionType.CARD and context == SelectContext.ATTACH_FROM:
            if "A" in ablate:
                return 0
            pk = get_card(obs, o.area, o.index, o.playerIndex)
            if isinstance(pk, Pokemon):
                if attacker_serial is not None and pk.serial == attacker_serial:
                    return 1500
                if pk.serial in plan.get("attach_serials", []):
                    return 1500
        elif o.type == OptionType.PLAY:
            if "P" in ablate:
                return 0
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            cid = card.id if card is not None else None
            seq = plan.get("trainer_seq", [])
            if cid is not None and cid in seq:
                rank = seq.index(cid)
                return max(200, 1200 - rank * 200)
        elif o.type == OptionType.RETREAT:
            if "R" in ablate:
                return 0
            if attacker_serial is not None:
                active = my_state.active[0] if my_state.active else None
                if active is not None and active.serial == attacker_serial:
                    return -4000
    except Exception:
        return 0
    return 0


def main_option_proc(obs: Observation, damage: int):
    state = obs.current
    select = obs.select
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]

    global can_switch
    global can_attack
    global can_main_attack
    global can_energy_attach

    can_switch = False
    can_attack = False
    can_main_attack = False
    can_energy_attach = False
    for o in select.option:
        if o.type == OptionType.RETREAT:
            can_switch = True
        elif o.type == OptionType.ATTACK:
            can_attack = True
            if o.attackId == 154:  # Phantom Dive
                can_main_attack = True
    
    plan_a.attack = -1
    plan_b.attack = -1
    if not can_main_attack and not (bench_attacker and can_switch):
        return
    
    cards = [op_state.active[0]]
    for pokemon in op_state.bench:
        cards.append(pokemon)
    counter_indices = []
    ci = []
    ci.append(0)
    remain_damage = 60
    while ci:
        index = ci[-1]
        hp = cards[index].hp
        if remain_damage >= hp:
            counter_indices.append(ci.copy())
            if index < len(cards) - 1:
                remain_damage -= hp
                ci.append(index + 1)
                continue
        if index == len(cards) - 1:
            ci.pop()
            if ci:
                remain_damage += cards[ci[-1]].hp
        if ci:
            ci[-1] += 1
    counter_indices.append([])

    remain_prize = len(my_state.prize)
    plan_score = 0
    for i, pokemon in enumerate(cards):
        base_prize_count = 0
        base_score = pokemon_score(pokemon, True)
        active_damage = 0 if no_damage_dex(pokemon.id) else damage
        if pokemon.hp <= active_damage:
            base_prize_count += prize_count(pokemon, True)
        else:
            base_score *= active_damage / pokemon.hp
        ci = []
        max_score = base_score
        if remain_prize <= base_prize_count:
            max_score = 50000
        else:
            for indices in counter_indices:
                if i in indices:
                    continue
                prize = base_prize_count
                score = base_score
                for index in indices:
                    prize += prize_count(cards[index], False)
                    score += pokemon_score(cards[index], False)
                if remain_prize <= prize:
                    score = 50000
                else:
                    if prize >= 2:
                        if remain_prize <= 4:
                            score -= 1200
                    elif prize == 1:
                        score -= 300
                    else:
                        score += 1200
                if max_score < score:
                    max_score = score
                    ci = indices
        if plan_score < max_score:
            plan_score = max_score
            plan_a.attack = i
            plan_a.counter = ci
        if i == 0:
            plan_b.attack = plan_a.attack
            plan_b.counter = plan_a.counter

def agent(obs_dict: dict) -> list[int]:
    """Main Agent Function.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount (inclusive), with no duplicate elements.
    
    Returns:
        list[int]: A list of option index.
    """
    obs = to_observation_class(obs_dict)
    if obs.select == None:
        # In the initial selection, the obs.select is None, and it is necessary to return the deck.
        # The deck is a list of 60 card IDs.
        # The deck must comply with the Pokémon Trading Card Game rules.
        return my_deck

    global pre_turn_log
    global current_turn_log
    global turn_plan

    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]
            
    if state.turn == 0:
        prize.clear()
        pre_turn_log.clear()
        current_turn_log.clear()
        turn_plan = None
    else:
        for log in obs.logs:
            current_turn_log.append(log)
            if log.type == LogType.TURN_END:
                pre_turn_log = current_turn_log
                current_turn_log = []

    pre_ko = False
    no_item = False
    for log in pre_turn_log:
        if log.type == LogType.ATTACK:
            if log.attackId == 323:  # Itchy Pollen
                no_item = True
        elif log.type == LogType.MOVE_CARD:
            if (log.playerIndex == my_index
                and (log.fromArea == AreaType.BENCH or log.fromArea == AreaType.ACTIVE)
                and log.toArea == AreaType.DISCARD):
                pre_ko = True

    if select.deck != None:
        set_card_counts(obs, my_index)
        for card in select.deck:
            card_counts[card.id] -= 1
        prize.clear()
        for id in card_counts:
            for _ in range(card_counts[id]):
                prize.append(id)
                
    set_card_counts(obs, my_index)
    for id in prize:
        card_counts[id] -= 1
    deck_counts = card_counts

    prize_diff = len(my_state.prize) - len(op_state.prize)
    
    global bench_attacker

    # Number of cards per card ID on the Bench and in the Active Spot
    field_counts = defaultdict(int)
    # Number of cards per card ID in hand
    hand_counts = defaultdict(int)
    # Number of cards per card ID in discard pile
    discard_counts = defaultdict(int)
    
    active_id = 0
    bench_attacker = False
    can_evolve_dreepy = False
    evolve_dreepy_count = 0
    can_evolve_drakloak = False
    damage = 200
    for card in my_state.active:
        if card == None:
            continue
        active_id = card.id
        field_counts[card.id] += 1
        if not card.appearThisTurn:
            if card.id == Dreepy:
                can_evolve_dreepy = True
                evolve_dreepy_count += 1
            elif card.id == Drakloak:
                can_evolve_drakloak = True
    for card in my_state.bench:
        field_counts[card.id] += 1
        if not card.appearThisTurn:
            if card.id == Dreepy:
                can_evolve_dreepy = True
                evolve_dreepy_count += 1
            elif card.id == Drakloak:
                can_evolve_drakloak = True
        if card.id == Dragapult_ex and len(card.energies) >= 2:
            bench_attacker = True
    main_pokemon_count = field_counts[Dreepy] + field_counts[Drakloak] + field_counts[Dragapult_ex]
    no_more_dex = (field_counts[Dragapult_ex] * 2 >= len(op_state.prize))

    stadium_id = 0
    for card in state.stadium:
        stadium_id = card.id

    support_count = 0

    for card in my_state.discard:
        discard_counts[card.id] += 1

    def attach_score(attach_id: int, pokemon: Pokemon, active: bool) -> int:
        energy_count = len(pokemon.energies)
        if card_table[attach_id].cardType == CardType.TOOL:
            # Attach tool
            score = 60000
            if active:
                score += 1000
            return score
        
        # Attach energy
        if pokemon.id == Budew:
            return -1
        elif pokemon.id == Meowth_ex or pokemon.id == Fezandipiti_ex or pokemon.id == Latias_ex:
            if active and not can_switch and not my_state.asleep and not my_state.paralyzed:
                if bench_attacker or field_counts[Budew] >= 1:
                    return 22000
                else:
                    return 18000
            else:
                return -1
        if active and can_main_attack:
            return -1
        score = 20000
        if energy_count >= 2:
            if active and not can_switch and not my_state.asleep and not my_state.paralyzed:
                score += 200
            else:
                return -1
        elif energy_count == 1:
            if attach_id == pokemon.energyCards[0].id:
                return -1
            if pokemon.id == Dragapult_ex:
                score += 250
            elif pokemon.id == Dreepy:
                score -= 150
            else:
                score -= 200
            if active:
                score += 200
        else:  # energy_count == 0
            if active:
                if bench_attacker:
                    score += 400
            else:
                if pokemon.id == Dragapult_ex:
                    score += 150
                elif pokemon.id == Dreepy:
                    score += 100
                else:
                    score += 50
                if bench_attacker:
                    score -= 200
        if no_more_dex and (pokemon.id == Dreepy or pokemon.id == Drakloak):
            score -= 500
        return score
    
    def hand_score(id: int, ignore_count: bool):
        score = 0
        if id == Dreepy:
            if main_pokemon_count >= 3:
                score = 1000
            else:
                score = 18000
        elif id == Drakloak:
            if can_evolve_dreepy:
                score = 20000
            else:
                score = 3000
        elif id == Dragapult_ex:
            if no_more_dex:
                score = UNNECESSARY
            elif can_evolve_dreepy and hand_counts[Rare_Candy] >= 1 and not no_item:
                score = 40000
            elif can_evolve_drakloak:
                if field_counts[id] == 0:
                    score = 30000
                elif field_counts[id] == 1:
                    score = 10000
                else:
                    score = 50
            else:
                if field_counts[id] >= 2:
                    score = 50
                else:
                    score = 2000
        elif id == Fezandipiti_ex:
            if pre_ko:
                score = 50000
            elif prize_diff <= -2:
                score = 5
            elif len(op_state.prize) == 1:
                score = UNNECESSARY
        elif id == Latias_ex:
            if active_id == Fezandipiti_ex or active_id == Meowth_ex or active_id == Dreepy:
                if field_counts[Drakloak] + field_counts[Dragapult_ex] == 0:
                    score = 28000
                else:
                    score = 15000
            else:
                score = 10
        elif id == Budew:
            if field_counts[id] + field_counts[Drakloak] + field_counts[Dragapult_ex] >= 1:
                score = UNNECESSARY
            elif state.turn >= 2:
                score = 30000
        elif id == Meowth_ex:
            if support_count > hand_counts[Boss_Orders] or stadium_id == Team_Rocket_Watchtower:
                score = 5
            elif state.supporterPlayed:
                score = 40
            else:
                score = 35000
        elif id == Rare_Candy:
            if no_more_dex:
                score = UNNECESSARY
            elif can_evolve_dreepy and hand_counts[Dragapult_ex] >= 1:
                score = 40000
        elif id == Unfair_Stamp:
            if pre_ko:
                score = 80000
            elif len(op_state.prize) == 1:
                score = UNNECESSARY
            else:
                score = 80
        elif id == Buddy_Buddy_Poffin:
            count = deck_counts[Dreepy]
            if count == 0:
                score = UNNECESSARY
            else:
                if state.turn <= 2 and field_counts[Budew] == 0 and deck_counts[Budew] >= 1:
                    count += 1
                if count >= 2:
                    score = 35000
        elif id == Night_Stretcher:
            for i in discard_counts:
                if discard_counts[i] >= 1:
                    card_type = card_table[i].cardType
                    if card_type == CardType.POKEMON or card_type == CardType.BASIC_ENERGY:
                        score = max(score, hand_score(i, ignore_count))
        elif id == Crushing_Hammer:
            score = 20
        elif id == Ultra_Ball:
            if main_pokemon_count <= 2 or field_counts[Dreepy] >= 1:
                score = 70
            else:
                score = 5
        elif id == Poke_Pad:
            score = max(hand_score(Dreepy, ignore_count), hand_score(Drakloak, ignore_count))
        elif id == Lucky_Helmet:
            score = 15
        elif id == Boss_Orders:
            if plan_a.attack > 0:
                score = 60000
        elif id == Crispin:
            if not ignore_count or support_count == 0:
                if deck_counts[Basic_Fire_Energy] == 0 or deck_counts[Basic_Psychic_Energy] == 0:
                    score = 10
                if not can_main_attack and not bench_attacker and field_counts[Dragapult_ex] >= 1:
                    score = 55000
                else:
                    score = 25000
        elif id == Brock_Scouting:
            if not ignore_count or support_count == 0:
                if state.turn == 2 and field_counts[Budew] + field_counts[Latias_ex] == 0:
                    score = 50000
                else:
                    score = 30000
        elif id == Lillie_Determination:
            if not ignore_count or support_count == 0:
                score = 45000
        elif id == Team_Rocket_Watchtower:
            if stadium_id != 0 and stadium_id != Team_Rocket_Watchtower:
                score = 4000
        elif id == Basic_Fire_Energy or id == Basic_Psychic_Energy:
            if can_main_attack and (len(op_state.prize) <= 2
                or (bench_attacker and len(op_state.prize) <= 4)):
                score = UNNECESSARY
            else:
                max_score = -10000
                for pokemon in my_state.active:
                    if pokemon == None:
                        continue
                    max_score = max(max_score, attach_score(id, pokemon, True))
                for pokemon in my_state.bench:
                    max_score = max(max_score, attach_score(id, pokemon, False))
                score = max_score - 5000
                if can_main_attack or bench_attacker:
                    score /= 10
        
        if not ignore_count and hand_counts[id] > 0:
            if id == Drakloak and hand_counts[id] < evolve_dreepy_count:
                score -= 10
            elif id == Dreepy:
                score -= 100
            else:
                score -= 100000
        return score

    global use_support
    if context == SelectContext.MAIN:
        main_option_proc(obs, damage)

        # --- Plan-then-subordinate: compute once per turn, revalidate after ---
        if turn_plan is None or turn_plan.get("turn") != state.turn:
            new_plan = None
            try:
                new_plan = compute_turn_plan(obs, my_index, deck_counts)
            except Exception:
                new_plan = None
            turn_plan = new_plan if new_plan is not None else {"turn": state.turn, "valid": False}
        else:
            try:
                if not _plan_feasible(turn_plan, obs, my_index):
                    turn_plan["valid"] = False
            except Exception:
                turn_plan["valid"] = False

        use_support = 0
        if not state.supporterPlayed:
            support_score = 0
            for o in select.option:
                if o.type == OptionType.PLAY:
                    card = get_card(obs, AreaType.HAND, o.index, state.yourIndex)
                    if card_table[card.id].cardType == CardType.SUPPORTER:
                        score = hand_score(card.id, True)
                        if support_score < score:
                            support_score = score
                            use_support = card.id

    hand_scores = []
    negative_hand_count = 0
    for card in my_state.hand:
        score = hand_score(card.id, False)
        hand_scores.append(score)
        if score < 0:
            negative_hand_count += 1
        hand_counts[card.id] += 1
        if card_table[card.id].cardType == CardType.SUPPORTER and card.id != Boss_Orders:
            support_count += 1

    no_draw = (my_state.deckCount <= 8)  # Whether to restrict actions that reduce the deck
    do_switch = (not can_main_attack and (bench_attacker or (active_id != Budew and field_counts[Budew] >= 1 and state.turn >= 2)))
    effect_card_id = 0 if select.effect == None else select.effect.id
    context_card_id = 0 if select.contextCard == None else select.contextCard.id

    active_plan = (turn_plan if (turn_plan is not None and turn_plan.get("valid", False)
                                  and turn_plan.get("turn") == state.turn) else None)

    scores = []  # Score for each action
    for o in select.option:
        score = 0  # The default and baseline score is 0.
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            if context == SelectContext.IS_FIRST:
                score = -1
            else:
                score = 1
        elif o.type == OptionType.CARD:
            card = get_card(obs, o.area, o.index, o.playerIndex)
            if card != None:
                energy_count = 0
                hp = 0
                if isinstance(card, Pokemon):
                    energy_count = len(card.energies)
                    hp = card.hp
                if (context == SelectContext.SWITCH
                    or context == SelectContext.TO_ACTIVE
                    or context == SelectContext.SETUP_ACTIVE_POKEMON):
                    # Selection of the Pokémon to send to the Active Spot
                    if o.playerIndex == my_index:
                        if card.id == Dreepy:
                            score += 10000
                        elif card.id == Drakloak:
                            if energy_count >= 1:
                                score += 20000
                            else:
                                score -= 10000
                        elif card.id == Dragapult_ex:
                            score += 50000
                        elif card.id == Budew:
                            if context != SelectContext.SWITCH:
                                score += 100000
                            elif not bench_attacker:
                                score += 30000
                        elif card.id == Fezandipiti_ex:
                            score -= 1000
                        elif card.id == Meowth_ex:
                            score -= 2000
                    else:
                        if plan_a.attack == o.index + 1:
                            score += 100000
                    score += energy_count * 1000
                    score += hp
                elif context == SelectContext.SETUP_BENCH_POKEMON:
                    if my_index == state.firstPlayer or card.id != Dreepy:
                        score = -1
                elif context == SelectContext.TO_BENCH or context == SelectContext.TO_HAND:
                    score = hand_score(card.id, False)
                    hand_counts[card.id] += 1
                    if effect_card_id == Crispin:
                        # Reverse scoring
                        score = 100000 - hand_score(card.id, True)
                elif context == SelectContext.DISCARD:
                    hand_counts[card.id] -= 1
                    if card_table[card.id].cardType == CardType.SUPPORTER:
                        support_count -= 1
                    score = -hand_score(card.id, False)
                elif context == SelectContext.DAMAGE_COUNTER or context == SelectContext.DAMAGE_COUNTER_ANY:
                    if hp > 0:
                        score = 100000 - 10 * hp + pokemon_score(card, False)
                        if context == SelectContext.DAMAGE_COUNTER:
                            if 210 <= hp <= 230:
                                score += 20000 + hp * 20
                                if o.area == AreaType.ACTIVE:
                                    score += 10000
                            elif 40 <= hp <= 90:
                                score += 10000 + hp * 20
                            elif hp <= 30:
                                score += -10000 + hp * 20
                            if card.id == 133 or card.id == 351:
                                score += 30000
                        else:
                            index = o.index + 1
                            if index in plan_b.counter:
                                score += 100000
                            else:
                                remain_damage = select.remainDamageCounter * 10
                                if 210 <= hp <= 200 + remain_damage:
                                    score += 30000
                                elif 20 <= hp <= 60 + remain_damage:
                                    score += 10000
                                elif hp == 10:
                                    score -= 100000
                            if no_damage_counter(card):
                                score = -1
                elif context == SelectContext.ATTACH_FROM:
                    score = attach_score(context_card_id, card, o.area == AreaType.ACTIVE)
                    if card.id == Dragapult_ex:
                        score += 200
        elif o.type == OptionType.ENERGY_CARD or o.type == OptionType.ENERGY:
            # Discarding energy (Retreat or Crushing Hammer)
            if o.playerIndex != state.yourIndex:
                if o.area == AreaType.BENCH:
                    score = 20
                else:
                    score = 10
                card = get_card(obs, o.area, o.index, o.playerIndex)
                if card_table[card.id].cardType == CardType.SPECIAL_ENERGY:
                    score += 1
        elif o.type == OptionType.PLAY:
            card = get_card(obs, AreaType.HAND, o.index, my_index)
            card_score = hand_scores[o.index]
            if card.id == Dreepy:
                score = 51000
            elif card.id == Fezandipiti_ex:
                if card_score > 0:
                    score = 53000
                else:
                    score = -1
            elif card.id == Latias_ex:
                if active_id != Drakloak and active_id != Dragapult_ex:
                    score = 51000
                else:
                    score = -1
            elif card.id == Budew:
                if field_counts[Budew] == 0 and field_counts[Dragapult_ex] == 0:
                    score = 52000
                else:
                    score = -1
            elif card.id == Meowth_ex:
                if state.supporterPlayed or stadium_id == Team_Rocket_Watchtower:
                    score = -1
                elif support_count == 0:
                    score = 50000
                elif support_count == hand_counts[Boss_Orders] and not plan_a.attack <= 0:
                    score = 50000
                else:
                    score = -1
            elif card.id == Rare_Candy:
                if no_more_dex:
                    score = -1
                else:
                    score = 75000
            elif card.id == Unfair_Stamp:
                score = 15000
            elif card.id == Night_Stretcher:
                if card_score >= 18000:
                    score = 42000
                else:
                    score = -1
            elif card.id == Crushing_Hammer:
                score = 40000
            elif card.id == Boss_Orders:
                if card.id == use_support:
                    score = 35000
                else:
                    score = -1
            elif card.id == Lillie_Determination:
                if card.id == use_support:
                    score = 14000
                else:
                    score = -1
            elif card.id == Team_Rocket_Watchtower:
                if stadium_id > 0 or state.turn == 1:
                    score = 80000
                else:
                    score = -1
            elif no_draw:
                score = -1
            elif card.id == Buddy_Buddy_Poffin:
                if deck_counts[Dreepy] > 0:
                    score = 46000
                else:
                    score = -1
            elif card.id == Ultra_Ball:
                if negative_hand_count >= 2:
                    score = 44000
                else:
                    score = -1
            elif card.id == Poke_Pad:
                if deck_counts[Dreepy] + deck_counts[Drakloak] > 0:
                    score = 45000
                else:
                    score = -1
            elif card.id == Crispin or card.id == Brock_Scouting:
                if card.id == use_support:
                    score = 35000
                else:
                    score = -1
        elif o.type == OptionType.ATTACH:
            card = get_card(obs, o.area, o.index, my_index)
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score = attach_score(card.id, pokemon, o.inPlayArea == AreaType.ACTIVE)
        elif o.type == OptionType.EVOLVE:
            pokemon = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
            score += len(pokemon.energies)
            if pokemon.id == Dreepy:
                score += 30000
            elif field_counts[Dragapult_ex] >= 2 or (field_counts[Dragapult_ex] == 1 and len(op_state.prize) <= 2):
                score = -1
            else:
                score += 70000
        elif o.type == OptionType.ABILITY:
            card = get_card(obs, o.area, o.index, my_index)
            if no_draw:
                score = -1
            elif card.id == 1267:  # Lumiose City
                score = 1
            else:
                score = 40000
        elif o.type == OptionType.RETREAT:
            if do_switch:
                score = 10000
            else:
                score = -1
        elif o.type == OptionType.ATTACK:
            score = o.attackId

        if active_plan is not None:
            try:
                score += _plan_score_adjustment(o, obs, context, my_index, my_state, active_plan)
            except Exception:
                pass

        scores.append(score)

    output = []
    if len(scores) >= 1:
        # Select in descending order of score
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        for i in range(select.maxCount):
            # If the score is negative, do not select it if skipping is possible
            if (sorted_scores[i][1] >= 0
                or select.minCount > i
                or (context != SelectContext.TO_BENCH and context != SelectContext.SETUP_BENCH_POKEMON)):
                output.append(sorted_scores[i][0])
                
    return output
