"""
Equity subpackage
──────────────────
Monte Carlo poker equity engine: a pure-Python (stdlib-only) hand evaluator,
a 169-hand preflop equity table with range modelling, and a range-aware
Monte Carlo simulator for hand win probabilities.

Ported and rewritten from dickreuter/neuron_poker's ``tools/`` (equity
calculator + hand evaluator + preflop table).
"""

from __future__ import annotations

from .cards import RANKS, SUITS, FULL_DECK, make_deck, two_card_notation
from .evaluator import (
    score_hand, hand_name, best_hand_index, HAND_NAMES,
    HIGH_CARD, PAIR, TWO_PAIR, THREE_OF_A_KIND, STRAIGHT,
    FLUSH, FULL_HOUSE, FOUR_OF_A_KIND, STRAIGHT_FLUSH,
)
from .preflop import PREFLOP_EQUITIES, top_range_hands
from .montecarlo import monte_carlo_equity, EquityResult

__all__ = [
    "RANKS", "SUITS", "FULL_DECK", "make_deck", "two_card_notation",
    "score_hand", "hand_name", "best_hand_index", "HAND_NAMES",
    "HIGH_CARD", "PAIR", "TWO_PAIR", "THREE_OF_A_KIND", "STRAIGHT",
    "FLUSH", "FULL_HOUSE", "FOUR_OF_A_KIND", "STRAIGHT_FLUSH",
    "PREFLOP_EQUITIES", "top_range_hands",
    "monte_carlo_equity", "EquityResult",
]
