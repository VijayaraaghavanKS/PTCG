import os
import time
import random
from collections import defaultdict, Counter

from cg.api import (AreaType, CardType, EnergyType, Observation, SelectContext, OptionType,
                     Card, Pokemon, all_card_data, all_attack, to_observation_class)
try:
    from cg.api import search_begin, search_step, search_end
    _SEARCH_IMPORTED = True
except Exception:
    _SEARCH_IMPORTED = False

"""
Bellibolt tech v1 -- Iono's Bellibolt ex, Electric type, + a 1-of Latias ex tech.

Forked from agent/bellibolt_v2 (belief-tracking + fusion-safe search Bellibolt
build). bellibolt_v2 (and its v1 base) have a real, repeatedly-confirmed
structural weak matchup: Mega Lucario ex-based decks (Fighting, Weakness
{P} Psychic, up to 340 HP) beat our all-Lightning attackers hard (18.8-40%
locally), because Bellibolt's own Pokemon are Weak to Fighting -- Lucario's
attacks get the weakness boost, ours never do. Mega Lucario ex is also the
single most common deck in our rating band (~28-29% share), so this one bad
matchup matters a lot.

DECK-LEVEL FIX (not decision-logic escalation, which was already tried and
found not to move this matchup -- see AGENT_LOG.md): swapped in exactly what
this deck was missing to fight back -- a single copy of Latias ex (184,
Basic Pokemon, 210 HP, Weakness {D} Darkness [irrelevant to Lucario],
Resistance {F} Fighting [-30 dmg from Lucario's own attacks], Eon Blade
{P}{P}(any) for 200 -> 400 with the Psychic-vs-Fighting weakness bonus, which
one-shots even a 340 HP Mega Lucario ex). Latias ex's own "Skyliner" ability
also gives all our Basic Pokemon free retreat while she's in play, a small
side bonus. Card-level changes from bellibolt_v2's deck.csv (60 cards, still
legal -- validated via lib/deck_validate.py):
  +1 Latias ex (184)              -- the tech attacker/answer
  -1 Canari (1233: 4 -> 3)        -- lowest-opportunity-cost trim available;
                                      still 3 copies of a Lightning-Pokemon
                                      tutor, the deck's other 3 draw/search
                                      engines (Poffin/Ultra Ball/Lillie's
                                      Determination) are untouched
  2x Basic {L} Energy -> 2x Basic {P} Energy (4 -> 20, new: 2x id 5)
                                   -- exactly enough to pay Eon Blade's two
                                      {P} pips (its 3rd pip is generic and can
                                      still be paid by our existing Lightning
                                      energy); nothing else in the deck can
                                      use Psychic energy, so these 2 cards
                                      have zero cost to the main engine except
                                      the 2 fewer Lightning energy in the pool.
Everything else (Pokemon lines, trainer counts, search/belief-tracking
architecture) is IDENTICAL to bellibolt_v2 -- this is a small, targeted patch,
not a rebuild.

MINIMAL decision-logic additions to support the new card (all additive, none
of the existing heuristic's core scoring paths were rewritten):
  - `_weakness_bonus_damage()`: generic weakness-aware damage estimate (mirrors
    the same technique validated in agent/latias_xerneas_v1), used to extend
    the existing Stage-1 "printed damage alone is already lethal" KO check so
    it also recognizes a weakness-doubled kill, not just a raw-damage one.
  - ATTACH scoring: routes Psychic energy specifically to Latias ex (nowhere
    else can use it) and blocks it from ever being wasted on the Lightning
    engine's colorless pips.
  - CARD search-target scoring (Ultra Ball's "fetch a Pokemon" step): small
    baseline priority on fetching Latias ex, boosted further when the belief
    model's confidence leans "lucario".
  - SWITCH/TO_ACTIVE scoring: promotes Latias ex to active when she's fully
    charged (2 Psychic + Eon Blade ready) and doing so would be lethal via the
    Psychic weakness bonus.
  - RETREAT (Stage 2, loss-shield): new highest-priority branch that retreats
    into Latias ex specifically when that promotion is a confirmed kill.

Everything below this point (belief tracking, staged heuristic, fusion-safe
search) is bellibolt_v2's own documentation, unchanged:

Clean-room upgrade of agent/iono_day1 (our proven 699.7-peak-rating ladder deck).
Same deck (deck.csv copied verbatim), same staged single-decision heuristic as the
base, restructured explicitly as:

    Stage 0  forced          -- only one legal option, take it, skip everything else.
    Stage 1  immediate-win/KO -- an ATTACK option whose printed base damage alone
                                  already KOs the opponent's active gets an
                                  overwhelming score (see heuristic_scores()).
    Stage 2  loss-shield      -- RETREAT/recovery-Trainer scoring biased by both a
                                  direct read of the opponent's active's attacks and
                                  the persistent archetype belief (technique 1).
    Stage 3  value-progress   -- the original iono_day1 per-option scoring, carried
                                  over essentially unchanged, with small belief-biased
                                  nudges on which attacker line to invest energy in.

On top of that staged heuristic sits an optional, fusion-safe shallow forward-search
layer (technique 2) built directly against cg.api's Search API (search_begin /
search_step / search_end). Its calling convention (sample a hidden-info
determinization, search_begin, search_step through candidate root actions, search_end)
was studied from the public 950+ LB notebook
research/public_notebooks/romanrozen_strong-start-baseline-agent-v10-lb-950 for
*mechanics only* -- everything below (belief model, candidate selection, rollout
policy, leaf evaluation, fusion-safety bookkeeping, margin gating) is our own
reasoning re-derived for this deck, not copied code.

Two techniques, both mandated by design:

1. Persistent opponent-archetype belief tracking (module-level state, updated once
   per real decision from cumulative revealed opponent cards -- see _update_belief).
   Feeds the plain heuristic on EVERY turn (Stage 2/3 above), independent of whether
   the search layer fires at all.

2. Fusion-safe shallow search (see _search_decide). "Fusion-safe" means: the
   candidate root actions are fixed ONCE from the plain heuristic's own ranking,
   BEFORE any hidden-info determinizations are sampled, and every candidate is
   evaluated against every sampled determinization; scores are then averaged per
   candidate. We never let an individual determinization pick its own independent
   best move and then vote on those picks -- that's the ISMCTS "strategy fusion"
   failure mode the assignment calls out, and this design structurally avoids it by
   construction (the loop is `for det: for candidate: evaluate`, never
   `for det: pick_best_for_this_det`).

The search is a pure optional enhancement: any failure at any point (import failure,
native API error, timeout, unexpected state) silently falls back to the plain
heuristic's own top pick. It never raises out of agent().
"""

# ==================== Deck (unchanged from iono_day1) ====================

file_path = "deck.csv"
if not os.path.exists(file_path):
    file_path = "/kaggle_simulations/agent/" + file_path
with open(file_path, "r") as file:
    csv = file.read().split("\n")
my_deck = []
for i in range(60):
    my_deck.append(int(csv[i]))

all_card = all_card_data()
card_table = {c.cardId: c for c in all_card}
attack_table = {a.attackId: a for a in all_attack()}

# Decklist (bellibolt_v2's decklist + the Latias ex tech patch -- see module
# docstring for the exact 3-card diff)
Iono_Voltorb = 265  # x3
Iono_Tadbulb = 268  # x3
Iono_Bellibolt_ex = 269  # x3
Iono_Wattrel = 270  # x3
Iono_Kilowattrel = 271  # x3
Latias_ex = 184  # x1 -- NEW: Fighting-resistant, Psychic-weakness-exploiting tech
Buddy_Buddy_Poffin = 1086  # x3
Night_Stretcher = 1097  # x2
Max_Rod = 1110  # x1
Energy_Retrieval = 1118  # x1
Ultra_Ball = 1121  # x3
Poke_Pad = 1152  # x2
Lillie_Determination = 1227  # x4
Canari = 1233  # x3 (trimmed from bellibolt_v2's x4 to make room)
Levincia = 1254  # x3
Basic_Lightning_Energy = 4  # x20 (trimmed from bellibolt_v2's x22)
Basic_Psychic_Energy = 5  # x2 -- NEW: exactly enough for Eon Blade's {P}{P} cost

INITIAL_PRIZE_COUNT = 6  # standard ruleset; deck sizes (60) confirm standard format


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


# ==================== Technique 1: opponent-archetype belief tracking ====================
# Reference decklists for our two known non-mirror sparring archetypes. Hardcoded
# here (read once, at import time, from what were agent/dragapult_day1/deck.csv and
# agent/mega_lucario_ref/deck.csv at the time this was written) since a Kaggle
# submission only ships this agent's own directory -- there is nothing to read at
# runtime. Bellibolt is included too so a mirror match gets identified as such.

_DRAGAPULT_DECK = [
    119, 119, 119, 119, 120, 120, 120, 120, 121, 121, 121, 140, 184, 235, 235,
    1071, 1079, 1079, 1080, 1086, 1086, 1086, 1086, 1097, 1097, 1120, 1120, 1120,
    1120, 1121, 1121, 1121, 1121, 1152, 1152, 1152, 1156, 1182, 1182, 1182, 1198,
    1198, 1198, 1198, 1210, 1210, 1227, 1227, 1227, 1227, 1256, 1256,
    2, 2, 2, 2, 5, 5, 5, 5,
]
_LUCARIO_DECK = [
    673, 673, 674, 674, 675, 675, 676, 676, 676, 677, 677, 677, 678, 678, 678, 678,
    1102, 1102, 1102, 1102, 1123, 1123, 1141, 1141, 1141, 1141, 1142, 1142, 1142,
    1142, 1152, 1152, 1152, 1152, 1159, 1182, 1182, 1192, 1192, 1192, 1192, 1227,
    1227, 1227, 1227, 1252, 1252,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
]
_ARCHETYPE_DECKS = {"dragapult": _DRAGAPULT_DECK, "lucario": _LUCARIO_DECK, "bellibolt": my_deck}


def _pokemon_ids_in(deck):
    return {cid for cid in set(deck) if card_table.get(cid) and card_table[cid].cardType == CardType.POKEMON}


_id_to_archs = defaultdict(set)
for _name, _deck in _ARCHETYPE_DECKS.items():
    for _cid in set(_deck):
        _id_to_archs[_cid].add(_name)

# Signature = card IDs unique to exactly one of the three known archetypes. This
# deliberately drops cards shared across 2+ decks (Poke Pad, Buddy-Buddy Poffin,
# Lillie's Determination, Boss's-Orders-equivalents, basic energy, ...) since those
# would otherwise dilute a count-based signal without telling us anything.
ARCHETYPE_SIGNATURE = {name: {cid for cid, archs in _id_to_archs.items() if archs == {name}}
                        for name in _ARCHETYPE_DECKS}
ARCHETYPE_SIGNATURE_POKEMON = {name: (sig & _pokemon_ids_in(_ARCHETYPE_DECKS[name]))
                                for name, sig in ARCHETYPE_SIGNATURE.items()}
ARCHETYPE_DECK_COUNT = {name: Counter(deck) for name, deck in _ARCHETYPE_DECKS.items()}

# Conservative "how hard can this archetype hit" prior, used as a floor for
# loss-shielding before we've directly observed the opponent's active's attack
# options (their known ex-attacker's printed damage). Kept modest/known-accurate:
# Dragapult ex's Phantom Dive = 200, Mega Lucario ex's Aura Jab = 130 (its cheaper,
# more repeatable attack -- Mega Brave's 270 needs 2 Fighting energy and a
# once-per-two-turns lockout, so 130 is the safer recurring-threat estimate),
# our own mirror's Thunderous Bolt = 230.
# NOTE: an earlier version of this used a flat per-archetype damage *floor* that
# got max()'d against the direct threat read. Empirically that over-triggered: as
# soon as any single Bellibolt-signature card was seen (near-immediate in a mirror
# match, e.g. the opponent's first Canari/Levincia in their discard) it forced
# `threat` up to 230 on every subsequent turn regardless of the opponent's actual
# current active Pokemon or its energy readiness, causing far too many premature
# "loss-shield" retreats and a losing (40% vs iono_day1 in an isolated ablation)
# record even before the search layer was involved. A margin *multiplier* on top
# of the directly-observed threat, gated to the two archetypes that actually run
# OHKO-capable ex attackers, fixed it (see `danger` below) -- belief still changes
# the decision in borderline cases, but only ever amplifies real, currently-visible
# information instead of substituting a constant worst-case regardless of context.
ARCHETYPE_DANGER_MARGIN = {"dragapult": 1.3, "lucario": 1.3}

# Most common Basic Pokemon per archetype -- used only to guess a face-down
# opponent active Pokemon's identity for search_begin (see _sample_hidden).
_ARCHETYPE_BASIC_GUESS = {}
for _name, _deck in _ARCHETYPE_DECKS.items():
    _counts = Counter(cid for cid in _deck
                       if card_table.get(cid) and card_table[cid].cardType == CardType.POKEMON
                       and card_table[cid].basic)
    _ARCHETYPE_BASIC_GUESS[_name] = _counts.most_common(1)[0][0] if _counts else Iono_Voltorb


def _archetype_basic_guess(dominant):
    return _ARCHETYPE_BASIC_GUESS.get(dominant, Iono_Voltorb)


# Persistent, per-match belief state. Reset on every new game (obs.select is None).
_belief_seen = Counter()   # card IDs ever observed from the opponent, cumulative
_belief_dominant = None    # best-guess archetype name, or None ("unknown")
_belief_score = 0.0        # raw evidence weight behind that guess


def _reset_belief():
    global _belief_seen, _belief_dominant, _belief_score
    _belief_seen = Counter()
    _belief_dominant = None
    _belief_score = 0.0


def _op_visible_ids(state, op_i):
    """Every card ID we can currently see belonging to the opponent."""
    op = state.players[op_i]
    seen = Counter()
    for p in (op.active + op.bench):
        if p is None:
            continue
        seen[p.id] += 1
        for c in p.energyCards:
            seen[c.id] += 1
        for c in p.tools:
            seen[c.id] += 1
        for c in p.preEvolution:
            seen[c.id] += 1
    for c in op.discard:
        seen[c.id] += 1
    for c in op.prize:
        if c is not None:
            seen[c.id] += 1
    if state.stadium and state.stadium[0].playerIndex == op_i:
        seen[state.stadium[0].id] += 1
    return seen


def _update_belief(state, my_index):
    """Simple count-based evidence accumulation -> dominant archetype guess.

    Only ever called on the real top-level observation with the real player's
    index, never from inside a search rollout (where obs.current.yourIndex flips
    to whichever side is deciding) -- that would corrupt the persistent belief by
    scanning our own deck as if it were "the opponent" whenever a rollout reaches
    an opponent decision. heuristic_scores() only *reads* the belief snapshot
    passed into it; it never re-derives it.
    """
    global _belief_seen, _belief_dominant, _belief_score
    op_i = 1 - my_index
    cur = _op_visible_ids(state, op_i)
    for cid, n in cur.items():
        if n > _belief_seen[cid]:
            _belief_seen[cid] = n

    scores = {}
    for name, sig in ARCHETYPE_SIGNATURE.items():
        pk = ARCHETYPE_SIGNATURE_POKEMON[name]
        cap = ARCHETYPE_DECK_COUNT[name]
        s = 0.0
        for cid in sig:
            n = min(_belief_seen.get(cid, 0), cap.get(cid, 0))
            if n <= 0:
                continue
            s += n * (2.0 if cid in pk else 1.0)  # Pokemon sightings count double
        scores[name] = s

    dominant = max(scores, key=scores.get)
    best = scores[dominant]
    if best <= 0:
        _belief_dominant, _belief_score = None, 0.0
    else:
        _belief_dominant, _belief_score = dominant, best
    return _belief_dominant, _belief_score


# ==================== Latias ex tech: weakness-aware damage helper ====================
# Generic, not hardcoded to Psychic-vs-Fighting specifically -- compares the
# ATTACKING Pokemon's own type (card_table[...].energyType) against the
# TARGET's printed Weakness (the correct TCG mechanic: weakness is a property
# of the attacker's type, not the energy that paid for the attack). Mirrors
# the technique already validated in agent/latias_xerneas_v1. Used only to
# ADD lethal-KO recognition on top of the existing plain-damage check in
# Stage 1 below -- never removes or weakens that check, so it can only turn a
# missed KO into a caught one, not the reverse.
def _weakness_bonus_damage(attacker_card_id, atk_damage, target_card_id):
    a = card_table.get(attacker_card_id)
    t = card_table.get(target_card_id)
    if a is None or t is None:
        return atk_damage
    if t.weakness is not None and t.weakness == a.energyType:
        return atk_damage * 2
    return atk_damage


# ==================== Staged heuristic scoring ====================

def heuristic_scores(obs: Observation, belief_dominant, belief_score):
    """Score every option in obs.select.option. Self-contained: recomputes all
    per-decision context from `obs` alone so it can be reused both for the real
    top-level decision and, generically, for either side's rollout moves inside
    the search layer (an unrecognized/foreign card ID just falls through to the
    generic default for its OptionType/CardType, which is a reasonable greedy
    proxy for "what would a sane player do here" -- same trick the reference
    notebook studied for mechanics relies on).
    """
    state = obs.current
    select = obs.select
    context = select.context
    my_index = state.yourIndex
    my_state = state.players[my_index]
    op_state = state.players[1 - my_index]
    op_prize = len(op_state.prize)
    my_prize_remaining = len(my_state.prize)

    field_counts = defaultdict(int)
    field_hand_counts = defaultdict(int)
    active_attacker = False
    bench_attacker = False

    energy_count = 0
    can_ability = False
    for p in my_state.active:
        if p is None:
            continue
        field_counts[p.id] += 1
        field_hand_counts[p.id] += 1
        energy_count += len(p.energies)
        if p.id == Iono_Kilowattrel and len(p.energies) > 0:
            can_ability = True
        if p.id == Iono_Voltorb and len(p.energies) >= 2:
            active_attacker = True
    for p in my_state.bench:
        field_counts[p.id] += 1
        field_hand_counts[p.id] += 1
        energy_count += len(p.energies)
        if p.id == Iono_Kilowattrel and len(p.energies) > 0:
            can_ability = True
        if p.id == Iono_Voltorb and len(p.energies) >= 2:
            bench_attacker = True
    field_pokemon1 = field_counts[Iono_Tadbulb] + field_counts[Iono_Bellibolt_ex]
    field_pokemon2 = field_counts[Iono_Wattrel] + field_counts[Iono_Kilowattrel]
    no_more_pokemon = (len(my_state.bench) >= 5)
    if field_counts[Iono_Tadbulb] + field_counts[Iono_Wattrel] >= 1:
        no_more_pokemon = False

    stadium_id = 0
    for c in state.stadium:
        stadium_id = c.id

    hand_counts = defaultdict(int)
    hand_scores = []
    unused_hand_count = 0
    for c in my_state.hand or []:
        score = -10000
        if c.id == Iono_Voltorb:
            score = 100
        elif c.id == Iono_Bellibolt_ex:
            if field_counts[c.id] <= 1:
                score = 120
        elif c.id == Iono_Kilowattrel:
            if field_counts[c.id] <= 1:
                score = 140
        elif c.id == Ultra_Ball:
            if not no_more_pokemon:
                score = 10
        elif c.id == Night_Stretcher:
            score = 50
        elif c.id == Energy_Retrieval:
            score = 20
        elif c.id == Max_Rod:
            score = 1000
        elif c.id == Lillie_Determination:
            score = 150
        elif c.id == Canari:
            score = 160
        elif c.id == Levincia:
            if stadium_id != Levincia:
                score = 30
        elif c.id == Basic_Lightning_Energy:
            score = -10
        elif c.id == Latias_ex:
            # Latias ex tech: without this, she falls through to the -10000
            # "worthless, discard me first" default that untracked cards get
            # -- a real bug found via testing that would have made Ultra
            # Ball/Canari's forced-discard costs throw our only copy of the
            # Lucario answer away before it's ever used. Protect her at a
            # tier comparable to our other named attackers.
            score = 140
        elif c.id == Basic_Psychic_Energy:
            # Same bug/fix as above: only 2 copies exist and Eon Blade is
            # useless without them, so they need explicit protection too
            # (well above Lightning Energy's -10 "plentiful, fine to
            # discard" treatment -- these 2 cards are scarce and load-bearing).
            score = 90
        score -= hand_counts[c.id] * 100
        hand_scores.append(score)
        if score < 0:
            unused_hand_count += 1
        hand_counts[c.id] += 1
        field_hand_counts[c.id] += 1

    discard_counts = defaultdict(int)
    for c in my_state.discard:
        discard_counts[c.id] += 1

    can_attack = False
    if context == SelectContext.MAIN:
        for o in select.option:
            if o.type == OptionType.ATTACK:
                can_attack = True

    op_active_hp = 10000
    op_active_pokemon = None
    if len(op_state.active) >= 1 and op_state.active[0] is not None:
        op_active_pokemon = op_state.active[0]
        op_active_hp = op_active_pokemon.hp

    # Latias ex tech: is she on our bench, fully charged (2 Psychic energy for
    # Eon Blade's {P}{P} pips + 1 more of any type for its generic 3rd pip),
    # and would promoting her right now be a confirmed kill via the Psychic
    # weakness bonus? Computed once here and reused by both the RETREAT
    # (Stage 2 loss-shield) and SWITCH/TO_ACTIVE scoring below.
    bench_latias_lethal = False
    if op_active_pokemon is not None:
        op_active_card = card_table.get(op_active_pokemon.id)
        if op_active_card is not None and op_active_card.weakness == EnergyType.PSYCHIC:
            for p in my_state.bench:
                if p.id != Latias_ex:
                    continue
                psy = sum(1 for e in p.energies if e == EnergyType.PSYCHIC)
                if psy >= 2 and len(p.energies) >= 3 and 200 * 2 >= op_active_hp:
                    bench_latias_lethal = True
                    break

    my_active = my_state.active[0] if (len(my_state.active) >= 1 and my_state.active[0] is not None) else None
    my_active_card = card_table.get(my_active.id) if my_active is not None else None
    active_is_valuable = bool(my_active_card and (my_active_card.ex or my_active_card.megaEx))

    no_draw = (my_state.deckCount <= 5)

    # --- Stage 2 (loss-shield) inputs: direct threat read + belief floor ---
    direct_threat = 0
    if op_active_pokemon is not None:
        oc = card_table.get(op_active_pokemon.id)
        if oc is not None:
            for aid in oc.attacks:
                a = attack_table.get(aid)
                if a is not None and a.damage > direct_threat:
                    direct_threat = a.damage
    danger_margin = ARCHETYPE_DANGER_MARGIN.get(belief_dominant, 1.0) if belief_score >= 3.0 else 1.0
    threat = direct_threat * danger_margin
    danger = (my_active is not None) and direct_threat > 0 and threat >= my_active.hp and active_is_valuable

    # Belief-biased nudges for which attacker line to invest in (Stage 3).
    lucario_lean = (belief_dominant == "lucario" and belief_score >= 2.0)
    dragapult_lean = (belief_dominant == "dragapult" and belief_score >= 2.0)

    prizes_lost = max(0, INITIAL_PRIZE_COUNT - my_prize_remaining)
    recovery_bump = 4000 if prizes_lost >= 2 else 0

    scores = []
    id_counts = defaultdict(int)
    for o in select.option:
        score = 0
        if o.type == OptionType.NUMBER:
            score = o.number
        elif o.type == OptionType.YES:
            score = 1
        elif o.type == OptionType.ATTACH or context == SelectContext.ATTACH_FROM:
            src = None
            if o.type == OptionType.ATTACH:
                p = get_card(obs, o.inPlayArea, o.inPlayIndex, my_index)
                src = get_card(obs, o.area, o.index, my_index)  # the energy card being attached
            else:
                p = get_card(obs, o.area, o.index, o.playerIndex)
            score = 40000
            # Latias ex tech: Psychic energy exists ONLY to pay Eon Blade's
            # {P}{P} cost -- nothing else in this deck can use it (Voltaic
            # Chain/Thunderous Bolt/Mach Bolt all explicitly reward {L} energy
            # specifically for their damage text, not colorless generically).
            # Route it to Latias ex and nowhere else; once her 2 pips are
            # filled, deprioritize further Psychic attaches (there are only 2
            # copies in the deck, so this is mostly moot, but keeps the score
            # honest if she somehow already has more).
            if src is not None and src.id == Basic_Psychic_Energy:
                if p.id == Latias_ex:
                    psy_have = sum(1 for e in p.energies if e == EnergyType.PSYCHIC)
                    score = 42000 if psy_have < 2 else 300
                else:
                    score = -20000
            elif p.id == Iono_Voltorb:
                if len(p.energies) >= 2:
                    if o.inPlayArea == AreaType.ACTIVE and not can_attack:
                        score += 3000
                else:
                    if o.inPlayArea == AreaType.ACTIVE:
                        score += 5000
                    elif bench_attacker or active_attacker:
                        score += 100
                    else:
                        score += 1000
            elif p.id == Iono_Tadbulb:
                score += 10 - len(p.energies)
            elif p.id == Iono_Bellibolt_ex:
                if len(p.energies) >= 4:
                    if o.inPlayArea == AreaType.ACTIVE and not can_attack:
                        score += 500
                else:
                    if o.inPlayArea == AreaType.ACTIVE:
                        score += 800
                    elif bench_attacker or active_attacker:
                        score += 14 - len(p.energies)
                    else:
                        score += 100
                    if lucario_lean:
                        # Technique 1 -> Stage 3: opponent looks like the tanky
                        # Mega Lucario ex line (up to 340 HP) -- lean harder into
                        # completing Bellibolt ex's full 230-damage line rather
                        # than spreading energy thin, since out-powering a tank
                        # matters more than racing width here.
                        score += 40
            elif p.id == Iono_Wattrel:
                if len(p.energies) >= 1 or o.inPlayArea == AreaType.ACTIVE:
                    score += 10 - len(p.energies)
                else:
                    score += 6000
                    if dragapult_lean:
                        score += 30
            elif p.id == Iono_Kilowattrel:
                if len(p.energies) >= 1:
                    score += 11 - len(p.energies)
                else:
                    score += 8000
                    if dragapult_lean:
                        # Technique 1 -> Stage 3: opponent looks like fast
                        # Dragapult ex aggro -- lean toward getting an attacker
                        # online quickly rather than maximally stacking one mon.
                        score += 30
            elif p.id == Latias_ex:
                # Latias ex tech: this branch only runs when the energy being
                # attached is NOT Psychic (the Psychic-source case is fully
                # handled above) -- i.e. it's one of our Lightning energy
                # cards. She can't use Lightning for anything (Eon Blade needs
                # 2 Psychic pips first; even its 3rd, generic pip is useless
                # to fill early). Bug found via A/B testing: leaving her at
                # the flat 40000 base let her silently outbid Tadbulb for
                # scarce early Lightning energy (Tadbulb's own bonus shrinks
                # toward 0 as it gains energy, so it can dip below Latias's
                # unmodified 40000) -- a real, previously-invisible tempo
                # leak that cost ~17 points of win rate vs dragapult_day1 in
                # testing. Pin her Lightning-attach priority to the bottom of
                # the range (still positive so a truly "nothing else to do"
                # turn can use her as a last resort) so she never competes
                # with the core engine for Lightning energy.
                score = 50
        elif o.type == OptionType.CARD:
            c = get_card(obs, o.area, o.index, o.playerIndex)
            if c is not None:
                data = card_table[c.id]
                if (context == SelectContext.SWITCH
                        or context == SelectContext.TO_ACTIVE
                        or context == SelectContext.SETUP_ACTIVE_POKEMON):
                    energy = 0
                    if isinstance(c, Pokemon):
                        energy = len(c.energies)
                        score -= c.hp
                        score -= energy * 100
                    if c.id == Iono_Voltorb:
                        if 20 + energy_count * 20 >= op_active_hp:
                            score += 100000
                        else:
                            score += 1500
                        if energy >= 1:
                            score += 200
                            if energy >= 2:
                                score += 10000
                    elif c.id == Iono_Bellibolt_ex:
                        score += 1000
                        if energy >= 4:
                            score += 1000
                    elif c.id == Iono_Tadbulb:
                        score += 10
                    elif c.id == Latias_ex:
                        # Latias ex tech: promoting her is only ever worth the
                        # generic HP/energy penalty above when it's a
                        # confirmed weakness-boosted kill on the opponent's
                        # current active (e.g. Mega Lucario ex) -- outweighs
                        # that penalty heavily so the promotion actually
                        # happens instead of being buried by her own high
                        # 210 HP/energy-investment cost.
                        if energy >= 3 and op_active_pokemon is not None:
                            psy = sum(1 for e in c.energies if e == EnergyType.PSYCHIC)
                            op_active_card = card_table.get(op_active_pokemon.id)
                            if (psy >= 2 and op_active_card is not None
                                    and op_active_card.weakness == EnergyType.PSYCHIC
                                    and 200 * 2 >= op_active_hp):
                                score += 200000
                elif context == SelectContext.TO_HAND or context == SelectContext.TO_BENCH:
                    if c.id == Basic_Lightning_Energy:
                        score += 1
                    elif c.id == Iono_Voltorb:
                        if o.area == AreaType.DISCARD:
                            score += 100000
                        if field_counts[c.id] == 0:
                            score += 110
                        elif field_counts[c.id] == 1 and op_prize >= 2:
                            score += 5
                    elif c.id == Iono_Tadbulb:
                        if field_pokemon1 == 0:
                            score += 200
                        elif field_pokemon1 == 1:
                            if op_prize >= 3 or (op_prize >= 2 and field_counts[Iono_Bellibolt_ex] == 0):
                                score += 20
                    elif c.id == Iono_Bellibolt_ex:
                        if field_hand_counts[c.id] == 0:
                            score += 250
                            if field_counts[Iono_Tadbulb] > 0:
                                score += 300
                        elif field_hand_counts[c.id] == 1:
                            if op_prize >= 3:
                                score += 30
                                if field_counts[Iono_Tadbulb] > 0:
                                    score += 30
                    elif c.id == Iono_Wattrel:
                        if field_pokemon2 == 0:
                            score += 320
                        elif field_pokemon2 == 1:
                            score += 15
                    elif c.id == Iono_Kilowattrel:
                        if field_hand_counts[c.id] == 0:
                            score += 300
                            if field_counts[Iono_Wattrel] > 0:
                                score += 250
                        elif field_hand_counts[c.id] == 1:
                            score += 25
                            if field_counts[Iono_Wattrel] > 0:
                                score += 25
                    elif c.id == Latias_ex:
                        # Latias ex tech: modest baseline priority for fetching
                        # (e.g. via Ultra Ball) our only answer to Fighting
                        # attackers even with no read on the opponent yet;
                        # much higher once belief confidently flags "lucario".
                        if field_counts[c.id] == 0:
                            score += 260 if lucario_lean else 15

                    if c.id != Basic_Lightning_Energy:
                        if hand_counts[c.id] >= 2:
                            score -= 20000
                        elif hand_counts[c.id] >= 1:
                            score -= 2000
                        if id_counts[c.id] == 1:
                            score -= 1000
                        elif id_counts[c.id] >= 2:
                            score -= 10000

                    id_counts[c.id] += 1
                elif context == SelectContext.DISCARD:
                    if o.area == AreaType.HAND and o.playerIndex == my_index and o.index < len(hand_scores):
                        score = -hand_scores[o.index]
        elif o.type == OptionType.PLAY:
            c = get_card(obs, AreaType.HAND, o.index, my_index)
            data = card_table[c.id]
            if data.cardType == CardType.STADIUM:
                if discard_counts[Basic_Lightning_Energy] >= 1 or can_ability:
                    score = 85000
                else:
                    score = -1
            elif data.cardType == CardType.SUPPORTER:
                score = 25000
                if c.id == Lillie_Determination:
                    score += 1000
                elif no_draw:
                    score = -1
                elif c.id == Canari:
                    if no_more_pokemon:
                        score = -1
                    elif field_counts[Iono_Voltorb] > 0 and field_counts[Iono_Bellibolt_ex] > 0 and field_counts[Iono_Kilowattrel] > 0:
                        score += 100
                    else:
                        score += 2000
            elif data.cardType == CardType.POKEMON:
                score = 100000
                if c.id == Iono_Voltorb and field_counts[Iono_Voltorb] >= 2:
                    score = -1
                elif c.id == Iono_Tadbulb and field_pokemon1 >= 2:
                    score = -1
                elif c.id == Iono_Wattrel and field_pokemon2 >= 2:
                    if op_prize >= 2 or field_counts[Iono_Voltorb] == 0 or field_counts[Iono_Bellibolt_ex] == 0:
                        score = -1
            else:
                if c.id == Night_Stretcher:
                    if (discard_counts[Iono_Voltorb] > 0
                            or (discard_counts[Iono_Bellibolt_ex] > 0 and field_counts[Iono_Tadbulb] > 0)
                            or (discard_counts[Iono_Kilowattrel] > 0 and field_counts[Iono_Wattrel] > 0)):
                        # Stage 2 (loss-shield): recover faster once we've already
                        # given up 2+ prizes -- rebuild board priority.
                        score = 75000 + recovery_bump
                    else:
                        score = -1
                elif c.id == Energy_Retrieval:
                    score = 61000
                elif c.id == Max_Rod:
                    if state.turn >= 3 and discard_counts[Basic_Lightning_Energy] >= 2:
                        score = 55000 + recovery_bump
                    else:
                        score = -1
                elif no_draw:
                    score = -1
                elif c.id == Buddy_Buddy_Poffin:
                    score = 80000
                elif c.id == Ultra_Ball:
                    if no_more_pokemon or state.turn <= 2:
                        score = -1
                    elif field_hand_counts[Iono_Bellibolt_ex] > 0 and field_hand_counts[Iono_Kilowattrel] > 0:
                        if unused_hand_count >= 2:
                            score = 45000
                        else:
                            score = -1
                    else:
                        if unused_hand_count >= 1:
                            score = 62000
                        else:
                            score = -1
                elif c.id == Poke_Pad:
                    score = 79000
        elif o.type == OptionType.EVOLVE:
            score = 110000
        elif o.type == OptionType.ABILITY:
            score = -1
            c = get_card(obs, o.area, o.index, my_index)
            if c.id == Iono_Bellibolt_ex:
                score = 50000
            elif c.id == Levincia:
                score = 8000
            elif not no_draw and c.id == Iono_Kilowattrel:
                score = 30000
        elif o.type == OptionType.RETREAT:
            if bench_latias_lethal:
                # Latias ex tech (Stage 2, highest loss-shield priority):
                # she's fully charged on the bench and promoting her is a
                # confirmed weakness-boosted kill (e.g. one-shotting a Mega
                # Lucario ex via Eon Blade) -- take the free kill over
                # whatever our current active would otherwise do this turn.
                score = 11000
            elif bench_attacker and not active_attacker:
                score = 10000
            elif danger and bench_attacker:
                # Stage 2 (loss-shield): technique 1's belief (or a direct read of
                # the opponent's revealed active's attacks) says a hit that would
                # KO our valuable (ex) active is likely incoming next turn, and we
                # have another attacker ready. Trade this turn's attack for
                # survival rather than risk a 2-prize KO.
                score = 8000
            else:
                score = -1
        elif o.type == OptionType.ATTACK:
            score = o.attackId  # unchanged baseline: keep attacking as a low-priority
            # fallback so setup/Trainer plays still happen first this turn, EXCEPT:
            atk = attack_table.get(o.attackId)
            if atk is not None and op_active_hp < 10000:
                real_dmg = atk.damage
                if my_active is not None and op_active_pokemon is not None:
                    # Latias ex tech: generalized weakness-aware damage check
                    # (see _weakness_bonus_damage) -- catches a weakness-
                    # doubled lethal (e.g. Eon Blade's 200 -> 400 vs a
                    # Psychic-weak target) that the plain printed-damage
                    # check below would miss. Still only ever a lower bound
                    # on real damage (weakness genuinely doubles whatever the
                    # engine's real total ends up being), so this can only
                    # turn a missed KO into a caught one, never a false
                    # positive.
                    real_dmg = _weakness_bonus_damage(my_active.id, atk.damage, op_active_pokemon.id)
                if real_dmg >= op_active_hp:
                    # Stage 1 (immediate-win/KO): the printed (or weakness-
                    # boosted) base damage alone is already lethal. Real damage
                    # (many attacks add "+X" text bonuses on top of the printed
                    # base) can only be >= this, so this check never produces a
                    # false positive -- only false negatives (attacks whose
                    # damage is entirely text-driven, e.g. Voltaic Chain's
                    # per-energy scaling, aren't caught here; the search layer
                    # can still find those via real engine simulation).
                    score = 900000 if my_prize_remaining <= 1 else 700000

        scores.append(score)

    return scores


# ==================== Technique 2: fusion-safe shallow forward search ====================

USE_SEARCH = _SEARCH_IMPORTED
TIME_BUDGET_S = 0.35          # per-decision budget, well under the reference's 0.8s
N_DET = 10                    # hidden-info determinizations per decision. Measured
                               # cost is only ~25-30ms per full _search_decide call
                               # (rollouts usually hand off within a couple of
                               # steps), far under the 350ms budget, so we can afford
                               # many more samples than the reference's 3 -- doing so
                               # matters a lot here: with only 3 samples, per-candidate
                               # averages were noisy enough to occasionally rank a
                               # heuristic-marked "avoid this" option (negative base
                               # score) above a legitimate one purely by chance.
MAX_CANDIDATES = 4            # fixed root candidates, chosen BEFORE sampling (fusion-safe)
SEARCH_MIN_OPTS = 2
SEARCH_MAX_OPTS = 16
MARGIN = 600.0                 # override threshold: ~0.6 prize's worth in leaf-eval units
                               # (leaf eval weights a prize swing at 1000). Tuned upward
                               # from an initial 400 after A/B testing showed 400 let
                               # through enough marginal/noisy overrides that the search
                               # layer was roughly a wash on top of the plain heuristic
                               # (~53% vs iono_day1 either way); 600 keeps the override
                               # rate lower and requires a clearer signal before trusting
                               # the N_DET-sample average over the proven plain heuristic.
MAX_OWN_ROLLOUT_STEPS = 15     # completes the rest of THIS turn only -- see
                               # _rollout_complete for why the opponent's following
                               # turn is deliberately not simulated
MATCH_SEARCH_TIME_CAP_MS = 90000.0   # hard per-match safety cap, far under the 600s
                                       # total agent budget even in a very long game
MAX_CONSECUTIVE_HARD_FAILURES = 5

_search_match_ms = 0.0
_search_consecutive_failures = 0
_search_disabled_this_match = not USE_SEARCH

_search_stats = {"calls": 0, "considered": 0, "overrides": 0, "hard_fail": 0, "total_ms": 0.0, "max_ms": 0.0}


def _reset_search_match_state():
    global _search_match_ms, _search_consecutive_failures, _search_disabled_this_match
    _search_match_ms = 0.0
    _search_consecutive_failures = 0
    _search_disabled_this_match = not USE_SEARCH


def _leaf_eval(state, my_index):
    if state is None:
        return 0.0
    if state.result is not None and state.result >= 0:
        if state.result == my_index:
            return 1_000_000.0
        if state.result == 2:
            return 0.0
        return -1_000_000.0
    me = state.players[my_index]
    op = state.players[1 - my_index]
    my_field = [p for p in (me.active + me.bench) if p]
    op_field = [p for p in (op.active + op.bench) if p]
    my_hp = sum(p.hp for p in my_field)
    op_hp = sum(p.hp for p in op_field)
    my_en = sum(len(p.energies) for p in my_field)
    op_en = sum(len(p.energies) for p in op_field)
    no_active_penalty = 0.0 if (me.active and me.active[0]) else 3000.0
    # len(x.prize) is x's OWN remaining prize cards to draw -- it goes DOWN as x
    # scores KOs. So (op.prize - me.prize) is high when we're ahead on KOs.
    return (1000.0 * (len(op.prize) - len(me.prize))
            + my_hp - op_hp
            + 5.0 * (my_en - op_en)
            - no_active_penalty)


def _rollout_pick(cur_obs, belief):
    sel = cur_obs.select
    if sel is None or len(sel.option) == 0:
        return None
    try:
        sc = heuristic_scores(cur_obs, belief[0], belief[1])
    except Exception:
        sc = list(range(len(sel.option), 0, -1))
    n = len(sel.option)
    order = sorted(range(n), key=lambda i: sc[i], reverse=True)
    k = min(sel.maxCount, n)
    k = max(k, min(max(1, sel.minCount), n))
    return order[:k]


def _rollout_complete(sid, cur, my_index, deadline, belief):
    """Greedily complete the REST OF OUR CURRENT TURN ONLY, then stop (as soon as
    the engine hands the turn to the opponent, or the step cap / time budget is
    hit). This is deliberately kept shallow: an earlier version also simulated the
    opponent's entire following turn via the same generic heuristic-proxy policy,
    but that added far more noise/bias than signal in practice -- with only a
    handful of samples, per-candidate leaf averages were dominated by how a mostly-
    unrelated (and, for foreign archetypes, largely default-scored) simulated
    opponent turn happened to play out, rather than by the actual merits of the
    candidate action itself. Measured empirically: adding more determinizations to
    average out that noise made win rate WORSE, not better (25% vs a 70% plain-
    heuristic baseline in the mirror matchup) -- a sign of systematic bias, not
    just variance, so the fix is a shorter horizon rather than more samples.
    Evaluating right at our own turn's end still captures real information the
    plain per-option heuristic can't see one step at a time (e.g. whether a KO
    lands, or which of two Trainer orderings leaves a better board by end of turn)
    while staying "shallow" as intended.
    """
    for _ in range(MAX_OWN_ROLLOUT_STEPS):
        if time.monotonic() > deadline:
            return sid, cur
        cs = cur.current
        if cs is None or (cs.result is not None and cs.result >= 0):
            return sid, cur
        if cur.select is None or cs.yourIndex != my_index:
            return sid, cur
        choice = _rollout_pick(cur, belief)
        if not choice:
            return sid, cur
        try:
            ss = search_step(sid, choice)
        except Exception:
            return sid, cur
        sid, cur = ss.searchId, ss.observation

    return sid, cur


def _sample_hidden(state, my_index, dominant, belief_score):
    """One determinization of all hidden zones (our own deck/facedown prizes, plus
    the opponent's deck/prize/hand/facedown-active). The opponent side is biased by
    technique 1's belief: once we have a dominant archetype guess, sample from that
    archetype's real 60-card decklist (minus what we've already seen them play);
    otherwise fall back to a generic pool built only from cards we've actually
    observed. Fixed root candidates are chosen by the caller BEFORE this is called
    for each determinization (see _search_decide) -- that ordering is what keeps
    this fusion-safe.
    """
    me = state.players[my_index]
    op_i = 1 - my_index
    op = state.players[op_i]

    seen_mine = Counter()
    for c in (me.hand or []):
        seen_mine[c.id] += 1
    for c in me.discard:
        seen_mine[c.id] += 1
    for c in me.prize:
        if c is not None:
            seen_mine[c.id] += 1
    for p in (me.active + me.bench):
        if p is None:
            continue
        seen_mine[p.id] += 1
        for c in p.energyCards:
            seen_mine[c.id] += 1
        for c in p.tools:
            seen_mine[c.id] += 1
        for c in p.preEvolution:
            seen_mine[c.id] += 1
    remain = []
    for cid, n in Counter(my_deck).items():
        left = n - seen_mine.get(cid, 0)
        if left > 0:
            remain.extend([cid] * left)
    n_prize_hidden = sum(1 for c in me.prize if c is None)
    need = me.deckCount + n_prize_hidden
    if len(remain) < need:
        remain += [Basic_Lightning_Energy] * (need - len(remain))
    random.shuffle(remain)
    your_deck = remain[:me.deckCount]
    fill = iter(remain[me.deckCount:need])
    your_prize = [c.id if c is not None else next(fill, Basic_Lightning_Energy) for c in me.prize]

    op_seen = _op_visible_ids(state, op_i)
    if dominant is not None and belief_score >= 1.0:
        cap = ARCHETYPE_DECK_COUNT[dominant]
        pool = []
        for cid, n in cap.items():
            left = n - op_seen.get(cid, 0)
            if left > 0:
                pool.extend([cid] * left)
    else:
        pool = []
        for cid, n in op_seen.items():
            pool.extend([cid] * n)
        if not pool:
            pool = [Basic_Lightning_Energy]

    n_op_prize_hidden = sum(1 for c in op.prize if c is None)
    op_need = op.deckCount + n_op_prize_hidden + op.handCount
    if len(pool) < op_need:
        pool += [Basic_Lightning_Energy] * (op_need - len(pool))
    random.shuffle(pool)
    opponent_deck = pool[:op.deckCount]
    off = op.deckCount
    fill_op = iter(pool[off:off + n_op_prize_hidden])
    opponent_prize = [c.id if c is not None else next(fill_op, Basic_Lightning_Energy) for c in op.prize]
    off += n_op_prize_hidden
    opponent_hand = pool[off:off + op.handCount]
    opponent_active = []
    if op.active and op.active[0] is None:
        opponent_active = [_archetype_basic_guess(dominant)]

    return dict(your_deck=your_deck, your_prize=your_prize,
                opponent_deck=opponent_deck, opponent_prize=opponent_prize,
                opponent_hand=opponent_hand, opponent_active=opponent_active)


def _search_decide(obs, order, scores, dominant, belief_score):
    """Return an overriding option index, or None to keep the plain heuristic pick.

    Fusion-safety: `candidates` is fixed once, from the plain heuristic's own
    ranking, before the determinization loop starts. Every candidate is then
    evaluated against every sampled determinization inside the SAME loop nest
    (for det: for candidate: ...) and scores are averaged per candidate at the end.
    No determinization is ever allowed to pick its own independent best move.
    """
    global _search_match_ms, _search_consecutive_failures, _search_disabled_this_match

    if _search_disabled_this_match:
        return None
    state = obs.current
    select = obs.select
    if state is None or select is None or select.context != SelectContext.MAIN:
        return None
    if select.maxCount != 1:
        return None
    n = len(select.option)
    if n < SEARCH_MIN_OPTS or n > SEARCH_MAX_OPTS:
        return None
    if state.turn < 2:
        return None
    if getattr(obs, "search_begin_input", None) is None:
        return None
    if _search_match_ms > MATCH_SEARCH_TIME_CAP_MS:
        return None

    my_index = state.yourIndex
    heur_top = order[0]
    if scores[heur_top] < 0:
        # The heuristic's own top pick is already in "nothing good to do" territory
        # (e.g. a near-degenerate decision point) -- not worth spending search
        # budget comparing among options it explicitly flagged as undesirable.
        return None
    if select.option[heur_top].type == OptionType.RETREAT and scores[heur_top] == 8000:
        # This is specifically the belief/direct-threat-driven loss-shield retreat
        # from Stage 2 (see heuristic_scores' `danger` branch), not the routine
        # "our active can't attack yet" retreat (score 10000). _rollout_complete is
        # deliberately shallow -- it only plays out the REST OF OUR CURRENT turn and
        # never simulates the opponent's reply -- so it structurally cannot see the
        # very thing this retreat exists to avoid (an incoming KO on our next
        # opponent turn). A/B testing vs dragapult_day1 confirmed the failure mode
        # directly: with search enabled it would sometimes override this retreat
        # with a play that only looks better by the end of our own turn, dropping
        # win rate from the heuristic-only 37.5% down to 18.8%. Trust the belief-
        # driven judgment here instead of second-guessing it with a horizon that
        # can't evaluate the risk being defended against.
        return None
    # Only ever compare candidates the plain heuristic considers plausible (score
    # >= 0). Options it explicitly marked as an "avoid this" sentinel (score < 0,
    # e.g. an item with no legal target right now) stay excluded even if they rank
    # in the top few purely because most other options are also negative here --
    # a noisy few-sample search average is not trustworthy evidence to override a
    # hand-tuned "don't do this".
    candidates = [i for i in order if scores[i] >= 0][:min(MAX_CANDIDATES, n)]
    if len(candidates) < 2:
        return None

    t0 = time.monotonic()
    deadline = t0 + TIME_BUDGET_S
    acc = {i: 0.0 for i in candidates}
    cnt = {i: 0 for i in candidates}
    belief = (dominant, belief_score)
    hard_failure = False

    try:
        for _det in range(N_DET):
            if time.monotonic() > deadline:
                break
            hidden = _sample_hidden(state, my_index, dominant, belief_score)
            try:
                ss0 = search_begin(obs, **hidden)
            except Exception:
                hard_failure = True
                break
            root_sid = ss0.searchId
            try:
                for idx in candidates:  # fixed candidate set, evaluated per-det, not chosen per-det
                    if time.monotonic() > deadline:
                        break
                    try:
                        ss = search_step(root_sid, [idx])
                    except Exception:
                        continue
                    sid, cur = ss.searchId, ss.observation
                    sid, cur = _rollout_complete(sid, cur, my_index, deadline, belief)
                    leaf = _leaf_eval(cur.current if cur is not None else None, my_index)
                    acc[idx] += leaf
                    cnt[idx] += 1
            finally:
                try:
                    search_end()
                except Exception:
                    pass
    except Exception:
        hard_failure = True
    finally:
        elapsed_ms = (time.monotonic() - t0) * 1000.0
        _search_match_ms += elapsed_ms
        _search_stats["calls"] += 1
        _search_stats["total_ms"] += elapsed_ms
        if elapsed_ms > _search_stats["max_ms"]:
            _search_stats["max_ms"] = elapsed_ms

    if hard_failure:
        _search_consecutive_failures += 1
        _search_stats["hard_fail"] += 1
        if _search_consecutive_failures >= MAX_CONSECUTIVE_HARD_FAILURES:
            _search_disabled_this_match = True
        return None
    _search_consecutive_failures = 0

    n_top = cnt.get(heur_top, 0)
    if n_top == 0:
        return None
    fair = [i for i in candidates if cnt[i] == n_top]  # only compare candidates sampled equally often
    if heur_top not in fair:
        return None
    avg = {i: acc[i] / cnt[i] for i in fair}
    best = max(fair, key=lambda i: avg[i])
    _search_stats["considered"] += 1
    if best == heur_top:
        return None
    if avg[best] < avg[heur_top] + MARGIN:
        return None
    _search_stats["overrides"] += 1
    return best


# ==================== Entry point ====================

def agent(obs_dict: dict) -> list[int]:
    """Main Agent Function.

    Each element in the returned list must be >= 0 and < len(obs.select.option).
    The list length must be between obs.select.minCount and obs.select.maxCount
    (inclusive), with no duplicate elements.
    """
    try:
        obs = to_observation_class(obs_dict)
    except Exception:
        return [0]

    if obs.select is None:
        _reset_belief()
        _reset_search_match_state()
        return my_deck

    select = obs.select
    n = len(select.option)
    if n == 0:
        return []

    # Stage 0: forced -- only one legal option, nothing to decide.
    if n == 1 and select.minCount <= 1 <= select.maxCount:
        return [0]

    state = obs.current
    my_index = state.yourIndex

    # Technique 1: refresh belief from the real observation only.
    dominant, belief_score = _update_belief(state, my_index)

    try:
        scores = heuristic_scores(obs, dominant, belief_score)
    except Exception:
        k = min(max(1, select.minCount), n)
        return list(range(k))

    order = sorted(range(n), key=lambda i: scores[i], reverse=True)
    fallback_pick = order[:select.maxCount]

    # Technique 2: fusion-safe shallow search, optional override of the top pick.
    if select.context == SelectContext.MAIN and select.maxCount == 1:
        try:
            override = _search_decide(obs, order, scores, dominant, belief_score)
        except Exception:
            override = None
        if override is not None:
            return [override]

    return fallback_pick
