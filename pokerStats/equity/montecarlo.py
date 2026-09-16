"""
Monte Carlo equity engine
──────────────────────────
Estimates a hand's equity (probability of winning, split pots counted
fractionally) against ``N`` opponents by dealing random runouts and scoring
showdowns with :mod:`pokerStats.equity.evaluator`.

Opponents can be modelled two ways, following the neuron_poker design:

* **random** — opponents are dealt uniformly from the remaining deck
  (``opponent_range=1.0``);
* **range-aware** — opponents are only dealt hole cards whose canonical
  starting hand is in the top-``opponent_range`` fraction of the preflop
  table (see :func:`pokerStats.equity.preflop.top_range_hands`), or in an
  explicit set of notation keys passed directly.

Pure Python, stdlib only; no ``treys``, no NumPy, no C++ extension. The hot
loop works on pre-encoded ``(rank, suit)`` integer tuples and uses the stdlib
``random`` module, which is markedly faster than per-card NumPy calls for the
small (~50-card) deck.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from pokerStats.equity.cards import FULL_DECK, RANKS, SUITS, two_card_notation
from pokerStats.equity.evaluator import score_ranks_suits
from pokerStats.equity.preflop import top_range_hands

# An opponent model is either a fraction of the range table, or an explicit
# set of starting-hand notation keys.
OpponentRange = Union[float, Iterable[str]]

BOARD_SIZE = 5
_MAX_RANGE_SAMPLE_TRIES = 200

# Pre-encode every card once as (rank_index, suit_index); the hot loop deals
# these tuples directly instead of re-parsing "As"-style strings each time.
_RANK_OF = {r: i for i, r in enumerate(RANKS)}
_SUIT_OF = {s: i for i, s in enumerate(SUITS)}
_ENCODED: Dict[str, Tuple[int, int]] = {
    c: (_RANK_OF[c[0]], _SUIT_OF[c[1]]) for c in FULL_DECK
}
Card = Tuple[int, int]


@dataclass
class EquityResult:
    """Outcome of a Monte Carlo equity run.

    ``equity`` = ``win_rate + tie_rate``, i.e. the hero's expected share of
    the pot (split pots already counted fractionally in ``tie_rate``).
    ``iterations`` is how many runouts were actually simulated.
    """

    equity: float
    win_rate: float
    tie_rate: float
    iterations: int


def _resolve_range(opponent_range: OpponentRange) -> Optional[Set[str]]:
    """Turn an opponent model into a set of allowed notation keys, or ``None``.

    ``None`` means "no restriction" (a fully random opponent), which lets the
    hot loop skip the range check entirely.
    """
    if isinstance(opponent_range, (int, float)):
        if opponent_range >= 1.0:
            return None
        return top_range_hands(float(opponent_range))
    return set(opponent_range)


def _range_as_cards(
    allowed_notation: Set[str], available: Sequence[str]
) -> List[Tuple[Card, Card]]:
    """Precompute the concrete (unordered) two-card combos in the range.

    Built once from ``available`` (the deck minus the hero's and board cards),
    so the sampler never has to reject a combo that uses an already-dead card.
    Returns each combo once as ``(a, b)`` with ``a < b``.
    """
    combos: List[Tuple[Card, Card]] = []
    cards = list(available)
    for i in range(len(cards)):
        for j in range(i + 1, len(cards)):
            if two_card_notation([cards[i], cards[j]]) in allowed_notation:
                a, b = _ENCODED[cards[i]], _ENCODED[cards[j]]
                combos.append((a, b) if a < b else (b, a))
    return combos


def monte_carlo_equity(
    hole_cards: Sequence[str],
    board: Sequence[str] = (),
    num_opponents: int = 1,
    iterations: int = 10_000,
    opponent_range: OpponentRange = 1.0,
    seed: Optional[int] = None,
) -> EquityResult:
    """Estimate the equity of ``hole_cards`` against ``num_opponents``.

    Parameters
    ----------
    hole_cards
        The hero's two cards, e.g. ``["As", "Ks"]``.
    board
        Known community cards (0..5). The rest are dealt each iteration.
    num_opponents
        Number of opposing players (each dealt two hole cards).
    iterations
        Number of random runouts to simulate.
    opponent_range
        ``1.0`` for random opponents, a smaller float for the top-X% range,
        or an explicit iterable of notation keys (e.g. ``{"AA", "AKs"}``).
    seed
        Optional seed for reproducible runs.
    """
    if len(hole_cards) != 2:
        raise ValueError("hole_cards must contain exactly two cards")
    if num_opponents < 1:
        raise ValueError("num_opponents must be >= 1")
    if len(board) > BOARD_SIZE:
        raise ValueError("board cannot exceed five cards")

    rng = random.Random(seed)
    allowed_notation = _resolve_range(opponent_range)

    known = set(hole_cards) | set(board)
    available = [c for c in FULL_DECK if c not in known]
    base_deck: List[Card] = [_ENCODED[c] for c in available]
    combo_list: Optional[List[Tuple[Card, Card]]] = None
    if allowed_notation is not None:
        combo_list = _range_as_cards(allowed_notation, available)
    hero = [_ENCODED[c] for c in hole_cards]
    board_enc = [_ENCODED[c] for c in board]
    to_deal = BOARD_SIZE - len(board_enc)
    cards_needed = 2 * num_opponents + to_deal

    wins = 0.0
    ties = 0.0
    for _ in range(iterations):
        # One draw supplies every unknown card this iteration; slicing it is
        # far cheaper than repeated pops.
        if combo_list is None:
            drawn = rng.sample(base_deck, cards_needed)
            opponents = [drawn[2 * k : 2 * k + 2] for k in range(num_opponents)]
            runout = board_enc + drawn[2 * num_opponents :]
        else:
            opponents, runout = _deal_with_ranges(
                rng, base_deck, num_opponents, board_enc, to_deal, combo_list
            )

        hero_ranks = [c[0] for c in hero] + [c[0] for c in runout]
        hero_suits = [c[1] for c in hero] + [c[1] for c in runout]
        best = score_ranks_suits(hero_ranks, hero_suits)

        beaten = False
        tied_with = 1  # hero
        for opp in opponents:
            opp_score = score_ranks_suits(
                [c[0] for c in opp] + [c[0] for c in runout],
                [c[1] for c in opp] + [c[1] for c in runout],
            )
            if opp_score > best:
                beaten = True
                break
            if opp_score == best:
                tied_with += 1

        if not beaten:
            if tied_with == 1:
                wins += 1.0
            else:
                ties += 1.0 / tied_with  # fractional share of a split pot

    win_rate = wins / iterations
    tie_rate = ties / iterations
    return EquityResult(
        equity=win_rate + tie_rate,
        win_rate=win_rate,
        tie_rate=tie_rate,
        iterations=iterations,
    )


def _deal_with_ranges(
    rng: random.Random,
    base_deck: Sequence[Card],
    num_opponents: int,
    board_enc: List[Card],
    to_deal: int,
    combo_list: Sequence[Tuple[Card, Card]],
) -> Tuple[List[List[Card]], List[Card]]:
    """Deal opponents by sampling directly from the allowed combo list.

    Sampling a combo from the precomputed list (rather than rejection-sampling
    the whole deck) makes tight ranges cheap: a single-hand range like ``KK``
    costs one draw instead of hundreds of misses. Each opponent's cards are
    then removed from the deck for the board runout; a combo colliding with an
    already-used card is re-drawn (bounded, with a random fallback).
    """
    used: Set[Card] = set()
    opponents: List[List[Card]] = []
    for _ in range(num_opponents):
        pair: Optional[List[Card]] = None
        for _try in range(_MAX_RANGE_SAMPLE_TRIES):
            a, b = combo_list[rng.randrange(len(combo_list))]
            if a not in used and b not in used:
                pair = [a, b]
                used.add(a)
                used.add(b)
                break
        if pair is None:  # range exhausted by prior deals — fall back to random
            for c in base_deck:
                if c not in used:
                    pair = pair + [c] if pair else [c]
                    used.add(c)
                    if len(pair) == 2:
                        break
        opponents.append(pair)  # type: ignore[arg-type]

    deck = [c for c in base_deck if c not in used]
    rng.shuffle(deck)
    runout = board_enc + deck[:to_deal]
    return opponents, runout
