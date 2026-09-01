"""
Plan-conditioned feature extraction: base_features.py's proven 138-dim
deck-agnostic vector, PLUS 6 additional "what's already been committed this
turn" features, computed from a TurnContextTracker that accumulates state
across the (causally-ordered) sequence of decisions within a single turn.

Motivation (see AGENT_LOG.md, 2026-08-13): 4 prior pure-BC attempts (bc_v1,
bc_v2, bc_v3_ko_override, bc_v4_ensemble) all failed as PRIMARY policy
(5-24% win rate) despite growing data volume and better lethal/KO features.
Diagnosis: the model is POINTWISE -- it scores each decision in total
isolation and can't reproduce turn-spanning sequencing (e.g. "don't attach
energy to Pokemon Y, we already committed to Pokemon X as this turn's
attacker three decisions ago"). These 6 features make that already-committed
context directly visible to the model, instead of forcing it to infer turn
history from a state snapshot that mostly already reflects it.

CRITICAL leak-safety property: the tracker is updated causally -- only after
a decision's OUTCOME (the action actually taken) is known, and only used to
inform LATER decisions in the same turn. It is never updated using
information from decisions that haven't happened yet, either offline (replay
extraction walks decisions in real chronological order) or online (a live
agent only ever knows its own past actions this turn). No lookahead.

FEATURE_DIM = 138 (base) + 6 (plan context) = 144.
"""
from cg.api import AreaType, OptionType, SelectContext, Pokemon

from base_features import extract_features as _base_extract_features, FEATURE_DIM as _BASE_DIM

N_PLAN = 6
FEATURE_DIM = _BASE_DIM + N_PLAN


class TurnContextTracker:
    """Per-match, per-turn accumulator of 'what have we already committed to
    this turn'. Reset at the start of every new turn. Call `plan_features(obs,
    opt)` BEFORE a decision is made (uses only state accumulated from EARLIER
    decisions this turn). Call `observe_chosen(obs, opt)` AFTER the decision's
    actual chosen option is known, to advance the accumulator for the next
    decision in the same turn."""

    def __init__(self):
        self._turn = None
        self._n_decisions = 0
        self._n_energy = 0
        self._n_evolve = 0
        self._n_trainer = 0
        self._attacked = False
        # serial -> count of attach/evolve investments this turn
        self._investment = {}

    def _maybe_reset(self, obs):
        turn = getattr(obs.current, "turn", None)
        if turn != self._turn:
            self._turn = turn
            self._n_decisions = 0
            self._n_energy = 0
            self._n_evolve = 0
            self._n_trainer = 0
            self._attacked = False
            self._investment = {}

    def _focus_serial(self):
        if not self._investment:
            return None
        return max(self._investment.items(), key=lambda kv: kv[1])[0]

    def _target_serial(self, obs, opt, my_index):
        """Best-effort serial of the Pokemon this option would invest in,
        for EVOLVE / ATTACH / ATTACH_FROM-context CARD options."""
        try:
            if opt.type == OptionType.EVOLVE or opt.type == OptionType.ATTACH:
                if opt.inPlayArea is None or opt.inPlayIndex is None:
                    return None
                ps = obs.current.players[my_index]
                area = ps.active if opt.inPlayArea == AreaType.ACTIVE else ps.bench
                pk = area[opt.inPlayIndex] if opt.inPlayIndex < len(area) else None
                return pk.serial if isinstance(pk, Pokemon) else None
            if opt.type == OptionType.CARD and obs.select is not None and obs.select.context == SelectContext.ATTACH_FROM:
                ps = obs.current.players[opt.playerIndex if opt.playerIndex is not None else my_index]
                area = ps.active if opt.area == AreaType.ACTIVE else ps.bench
                pk = area[opt.index] if (area is not None and opt.index is not None and opt.index < len(area)) else None
                return pk.serial if isinstance(pk, Pokemon) else None
        except Exception:
            return None
        return None

    def plan_features(self, obs, opt) -> list:
        self._maybe_reset(obs)
        my_index = obs.current.yourIndex
        f0 = min(self._n_decisions, 10) / 10.0
        f1 = min(self._n_energy, 4) / 4.0
        f2 = min(self._n_evolve, 3) / 3.0
        f3 = min(self._n_trainer, 6) / 6.0
        f4 = 1.0 if self._attacked else 0.0
        f5 = 0.0
        try:
            focus = self._focus_serial()
            if focus is not None:
                tgt = self._target_serial(obs, opt, my_index)
                if tgt is not None and tgt == focus:
                    f5 = 1.0
        except Exception:
            f5 = 0.0
        return [f0, f1, f2, f3, f4, f5]

    def observe_chosen(self, obs, opt):
        """Advance the accumulator using the option that was ACTUALLY chosen
        for the just-completed decision. Safe to call with try/except by
        caller; internal exceptions are swallowed (never blocks play)."""
        self._maybe_reset(obs)
        my_index = obs.current.yourIndex
        self._n_decisions += 1
        try:
            if opt.type == OptionType.ATTACK:
                self._attacked = True
            elif opt.type == OptionType.EVOLVE:
                self._n_evolve += 1
                s = self._target_serial(obs, opt, my_index)
                if s is not None:
                    self._investment[s] = self._investment.get(s, 0) + 2
            elif opt.type == OptionType.ATTACH:
                self._n_energy += 1
                s = self._target_serial(obs, opt, my_index)
                if s is not None:
                    self._investment[s] = self._investment.get(s, 0) + 1
            elif opt.type == OptionType.CARD and obs.select is not None and obs.select.context == SelectContext.ATTACH_FROM:
                self._n_energy += 1
                s = self._target_serial(obs, opt, my_index)
                if s is not None:
                    self._investment[s] = self._investment.get(s, 0) + 1
            elif opt.type == OptionType.PLAY:
                card = None
                try:
                    ps = obs.current.players[my_index]
                    card = ps.hand[opt.index] if (ps.hand and opt.index is not None and opt.index < len(ps.hand)) else None
                except Exception:
                    card = None
                if card is not None:
                    self._n_trainer += 1
        except Exception:
            pass


def extract_features(obs, opt, card_table, attack_table, tracker: "TurnContextTracker" = None) -> list:
    base = _base_extract_features(obs, opt, card_table, attack_table)
    if tracker is None:
        plan = [0.0] * N_PLAN
    else:
        try:
            plan = tracker.plan_features(obs, opt)
        except Exception:
            plan = [0.0] * N_PLAN
    assert len(plan) == N_PLAN
    out = base + plan
    assert len(out) == FEATURE_DIM
    return out
