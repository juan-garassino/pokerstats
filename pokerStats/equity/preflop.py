"""
Precomputed preflop hand equities (169 canonical starting hands)
────────────────────────────────────────────────────────────────
Each value is the heads-up equity of the starting hand versus a uniformly
random single opponent over a full runout, taken from the reference table in
dickreuter/neuron_poker ``tools/montecarlo_python.py`` (``preflop_equities``).

Keys use pokerStats' two-card short notation (see
:func:`pokerStats.equity.cards.two_card_notation`): rank-ordered high→low,
suffixed ``s`` (suited) / ``o`` (offsuit), pairs unsuffixed — e.g. ``"AKs"``,
``"AKo"``, ``"AA"``. The upstream neuron_poker table stored non-pair keys
**low-rank-first** (``"KAs"``, ``"2To"``) with uppercase ``S``/``O`` suffixes;
both are normalised on port — ranks reordered high→low and suffixes lowercased —
so every key matches what ``two_card_notation`` emits. (Getting this wrong once
made range-aware equity silently drop all non-pair hands, since only the 13
pocket-pair keys are identical under either ordering.)

Use :func:`top_range_hands` to build an opponent hand range (the top-X%
strongest starting hands), which the Monte Carlo engine samples opponents
from for range-aware equity.
"""

from __future__ import annotations

from typing import Set

# 169 hands. Values are win probabilities heads-up vs a random hand.
# Keys are canonical high-rank-first notation (see the module docstring); the
# upstream neuron_poker table stored them low-rank-first, so they are reordered
# on port to match :func:`pokerStats.equity.cards.two_card_notation`.
PREFLOP_EQUITIES = {
    "32o": 0.354, "42o": 0.362, "62o": 0.376, "43o": 0.379, "52o": 0.380,
    "72o": 0.381, "32s": 0.388, "63o": 0.390, "53o": 0.390, "73o": 0.396,
    "83o": 0.400, "42s": 0.402, "72s": 0.405, "82o": 0.406, "52s": 0.408,
    "62s": 0.410, "64o": 0.411, "74o": 0.417, "43s": 0.418, "54o": 0.418,
    "92o": 0.422, "84o": 0.423, "73s": 0.424, "53s": 0.425, "63s": 0.427,
    "93o": 0.427, "65o": 0.428, "82s": 0.429, "75o": 0.433, "94o": 0.435,
    "83s": 0.436, "T2o": 0.436, "85o": 0.438, "54s": 0.443, "64s": 0.444,
    "74s": 0.445, "76o": 0.446, "T3o": 0.453, "84s": 0.453, "65s": 0.454,
    "92s": 0.455, "93s": 0.457, "86o": 0.457, "95o": 0.459, "75s": 0.464,
    "T4o": 0.465, "94s": 0.465, "J2o": 0.466, "96o": 0.467, "T5o": 0.467,
    "T2s": 0.468, "87o": 0.472, "85s": 0.476, "T6o": 0.477, "T3s": 0.477,
    "97o": 0.479, "76s": 0.481, "J3o": 0.481, "95s": 0.481, "86s": 0.483,
    "J4o": 0.486, "Q2o": 0.493, "J2s": 0.494, "96s": 0.495, "Q3o": 0.497,
    "T4s": 0.497, "J6o": 0.497, "J5o": 0.498, "T5s": 0.501, "T7o": 0.502,
    "98o": 0.504, "T6s": 0.505, "87s": 0.506, "J3s": 0.512, "T8o": 0.513,
    "J7o": 0.515, "J4s": 0.517, "97s": 0.518, "T7s": 0.519, "Q2s": 0.519,
    "22": 0.519, "Q4o": 0.521, "J6s": 0.522, "Q5o": 0.525, "J5s": 0.525,
    "K2o": 0.528, "K3o": 0.528, "98s": 0.529, "J8o": 0.530, "T9o": 0.530,
    "Q6o": 0.532, "Q4s": 0.533, "Q3s": 0.535, "Q7o": 0.536, "J7s": 0.539,
    "K4o": 0.539, "T8s": 0.539, "K5o": 0.541, "J9o": 0.547, "Q5s": 0.548,
    "K2s": 0.551, "33": 0.556, "T9s": 0.556, "Q8o": 0.557, "Q6s": 0.557,
    "J8s": 0.559, "Q7s": 0.560, "K6o": 0.560, "K4s": 0.564, "Q9o": 0.565,
    "K3s": 0.565, "A2o": 0.567, "K8o": 0.567, "JTo": 0.568, "K7o": 0.568,
    "A3o": 0.574, "Q8s": 0.575, "K5s": 0.576, "J9s": 0.580, "44": 0.582,
    "K6s": 0.585, "QTo": 0.586, "A2s": 0.588, "Q9s": 0.588, "K7s": 0.590,
    "K9o": 0.591, "QJo": 0.591, "A4o": 0.592, "A6o": 0.593, "A5o": 0.594,
    "JTs": 0.594, "K8s": 0.596, "A7o": 0.600, "A3s": 0.601, "QTs": 0.606,
    "KTo": 0.606, "A4s": 0.607, "A9o": 0.608, "QJs": 0.610, "KJo": 0.611,
    "K9s": 0.615, "A8o": 0.616, "55": 0.619, "A6s": 0.620, "KQo": 0.624,
    "A5s": 0.626, "A7s": 0.628, "A8s": 0.631, "KTs": 0.635, "ATo": 0.637,
    "66": 0.640, "KQs": 0.641, "A9s": 0.643, "KJs": 0.644, "AJo": 0.646,
    "ATs": 0.650, "AQo": 0.651, "77": 0.659, "AKo": 0.659, "AJs": 0.661,
    "AQs": 0.667, "AKs": 0.682, "88": 0.698, "99": 0.720, "TT": 0.752,
    "JJ": 0.775, "QQ": 0.802, "KK": 0.831, "AA": 0.853,
}


def top_range_hands(fraction: float) -> Set[str]:
    """Return the notation keys for the strongest ``fraction`` of hands.

    ``fraction=1.0`` returns all 169 hands (a fully random opponent);
    ``fraction=0.1`` returns roughly the top-10% strongest starting hands.
    The result is a set of keys such as ``{"AA", "KK", "AKs", ...}`` for the
    Monte Carlo engine to sample opponent hole cards from.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    ordered = sorted(PREFLOP_EQUITIES, key=PREFLOP_EQUITIES.get, reverse=True)
    take = max(1, round(len(ordered) * fraction))
    return set(ordered[:take])
