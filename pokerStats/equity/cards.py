"""
Card conventions for the equity engine
──────────────────────────────────────
A card is a two-character string ``"<rank><suit>"`` matching the rest of
pokerStats: rank in ``"23456789TJQKA"``, suit in ``"cdhs"`` (lowercase, as
used by :mod:`pokerStats.rl.poker_env` and the ``treys`` library).

This module owns the deck vocabulary plus the "two-card short notation" used
to key the preflop range table (e.g. ``"AKs"``, ``"AKo"``, ``"AA"``).
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

RANKS: str = "23456789TJQKA"
SUITS: str = "cdhs"
FULL_DECK: Tuple[str, ...] = tuple(f"{r}{s}" for r in RANKS for s in SUITS)


def rank_index(card: str) -> int:
    """Return the 0-based rank strength of ``card`` (2→0, …, A→12)."""
    return RANKS.index(card[0])


def make_deck(exclude: Iterable[str] = ()) -> List[str]:
    """Return a fresh 52-card deck minus any cards in ``exclude``.

    ``exclude`` typically holds the hero's hole cards and the known board so
    they are never dealt twice during a simulation.
    """
    dead = set(exclude)
    return [c for c in FULL_DECK if c not in dead]


def two_card_notation(cards: Iterable[str]) -> str:
    """Collapse two hole cards to their preflop-range key.

    The key is rank-ordered high→low and suffixed ``s`` (suited) or ``o``
    (offsuit); pocket pairs get no suffix. So ``["Ah", "Ks"]`` → ``"AKo"``,
    ``["Ah", "Kh"]`` → ``"AKs"``, ``["Ah", "Ac"]`` → ``"AA"``. This matches
    the keys in :data:`pokerStats.equity.preflop.PREFLOP_EQUITIES`.
    """
    c0, c1 = list(cards)
    (hi, lo) = (c0, c1) if rank_index(c0) >= rank_index(c1) else (c1, c0)
    if hi[0] == lo[0]:
        return hi[0] + lo[0]  # pocket pair, no suit suffix
    suffix = "s" if c0[1] == c1[1] else "o"
    return f"{hi[0]}{lo[0]}{suffix}"
