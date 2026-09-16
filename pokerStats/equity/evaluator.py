"""
Pure-Python 7-card hand evaluator
──────────────────────────────────
Scores a Texas Hold'em hand (2 hole + up to 5 board = 5..7 cards) into a
comparable tuple, so ``score(a) > score(b)`` iff hand ``a`` beats hand ``b``.

Ported and rewritten from dickreuter/neuron_poker ``tools/hand_evaluator.py``.
Kept dependency-free (no ``treys``) so the equity engine can run anywhere and
so ties are resolved by full kicker comparison rather than treys' rank buckets.

Rank strength uses the 0-based index into ``"23456789TJQKA"`` (Ace high, with
a wheel adjustment for A-5 straights). The hot path (:func:`score_ranks_suits`)
operates on pre-parsed integer rank/suit lists so the Monte Carlo loop never
re-parses card strings.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Sequence, Tuple

from pokerStats.equity.cards import RANKS, SUITS

# Category rank of each hand class, high is better. Used as the leading
# element of the score tuple so category dominates before kickers.
HIGH_CARD = 0
PAIR = 1
TWO_PAIR = 2
THREE_OF_A_KIND = 3
STRAIGHT = 4
FLUSH = 5
FULL_HOUSE = 6
FOUR_OF_A_KIND = 7
STRAIGHT_FLUSH = 8

HAND_NAMES = {
    HIGH_CARD: "HighCard",
    PAIR: "Pair",
    TWO_PAIR: "TwoPair",
    THREE_OF_A_KIND: "ThreeOfAKind",
    STRAIGHT: "Straight",
    FLUSH: "Flush",
    FULL_HOUSE: "FullHouse",
    FOUR_OF_A_KIND: "FourOfAKind",
    STRAIGHT_FLUSH: "StraightFlush",
}

# Score is (category, tie-breaker rank tuple). Both are comparable, and the
# outer tuple compares category first then kickers, which is exactly the
# poker ordering.
Score = Tuple[int, Tuple[int, ...]]

_SUIT_INDEX = {s: i for i, s in enumerate(SUITS)}


def _straight_high(ranks: set) -> int:
    """Return the high card of the best straight in ``ranks``, else ``-1``.

    ``ranks`` is a set of distinct 0-based ranks. The Ace (index 12) is also
    treated as index ``-1`` so it can complete the A-2-3-4-5 wheel.
    """
    if 12 in ranks:
        ranks = ranks | {-1}  # wheel: Ace plays low
    for high in sorted(ranks, reverse=True):
        # A straight is five consecutive ranks descending from ``high``.
        if high - 4 in ranks and high - 3 in ranks and high - 2 in ranks and high - 1 in ranks:
            return high
    return -1


def score_ranks_suits(ranks: Sequence[int], suits: Sequence[int]) -> Score:
    """Score a hand from pre-parsed integer rank/suit lists (the hot path).

    ``ranks`` are 0-based (2→0 … A→12) and ``suits`` are 0..3. This is the
    core routine; :func:`score_hand` is a string-parsing convenience wrapper.
    """
    rank_counts = Counter(ranks)
    # Order ranks by (count, rank) so pairs/trips sort ahead of their kickers.
    by_strength = sorted(rank_counts, key=lambda r: (rank_counts[r], r), reverse=True)
    count_pattern = sorted(rank_counts.values(), reverse=True)

    # Flush: a suit appearing >= 5 times. Keep its five highest ranks.
    suit_counts = Counter(suits)
    flush_suit = next((s for s, c in suit_counts.items() if c >= 5), None)
    flush_ranks: List[int] = []
    if flush_suit is not None:
        flush_ranks = sorted(
            (r for r, s in zip(ranks, suits) if s == flush_suit), reverse=True
        )

    unique_ranks = set(ranks)
    straight_high = _straight_high(unique_ranks)

    # Straight flush: a straight entirely within the flush suit.
    if flush_suit is not None:
        sf_high = _straight_high(set(flush_ranks))
        if sf_high >= 0:
            return (STRAIGHT_FLUSH, (sf_high,))

    if count_pattern[0] == 4:
        quad = by_strength[0]
        kicker = max(r for r in ranks if r != quad)
        return (FOUR_OF_A_KIND, (quad, kicker))

    if count_pattern[0] == 3 and len(count_pattern) > 1 and count_pattern[1] >= 2:
        trips = by_strength[0]
        pair = next(r for r in by_strength[1:] if rank_counts[r] >= 2)
        return (FULL_HOUSE, (trips, pair))

    if flush_suit is not None:
        return (FLUSH, tuple(flush_ranks[:5]))

    if straight_high >= 0:
        return (STRAIGHT, (straight_high,))

    if count_pattern[0] == 3:
        trips = by_strength[0]
        kickers = tuple(r for r in by_strength if r != trips)[:2]
        return (THREE_OF_A_KIND, (trips,) + kickers)

    if count_pattern[0] == 2 and len(count_pattern) > 1 and count_pattern[1] == 2:
        pairs = tuple(r for r in by_strength if rank_counts[r] == 2)[:2]
        kicker = max(r for r in ranks if r not in pairs)
        return (TWO_PAIR, pairs + (kicker,))

    if count_pattern[0] == 2:
        pair = by_strength[0]
        kickers = tuple(r for r in by_strength if r != pair)[:3]
        return (PAIR, (pair,) + kickers)

    return (HIGH_CARD, tuple(sorted(ranks, reverse=True)[:5]))


def score_hand(cards: Sequence[str]) -> Score:
    """Return the best 5-card :data:`Score` from 5..7 card strings.

    Cards are two-char strings (``"As"``); suits are matched
    case-insensitively so both the lowercase pokerStats convention and any
    uppercase input work.
    """
    ranks = [RANKS.index(c[0]) for c in cards]
    suits = [_SUIT_INDEX[c[1].lower()] for c in cards]
    return score_ranks_suits(ranks, suits)


def hand_name(score: Score) -> str:
    """Human-readable category name for a :data:`Score`."""
    return HAND_NAMES[score[0]]


def best_hand_index(hands: Sequence[Sequence[str]]) -> Tuple[List[int], Score]:
    """Return the indices of the winning hand(s) and the winning score.

    Ties (equal scores) return every winning index, so equity can split the
    pot fractionally rather than mis-crediting a single seat.
    """
    scores = [score_hand(h) for h in hands]
    best = max(scores)
    winners = [i for i, s in enumerate(scores) if s == best]
    return winners, best
