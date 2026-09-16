"""
Precomputed preflop hand equities (169 canonical starting hands)
────────────────────────────────────────────────────────────────
Each value is the heads-up equity of the starting hand versus a uniformly
random single opponent over a full runout, taken from the reference table in
dickreuter/neuron_poker ``tools/montecarlo_python.py`` (``preflop_equities``).

Keys use pokerStats' two-card short notation (see
:func:`pokerStats.equity.cards.two_card_notation`): rank-ordered high→low,
suffixed ``s`` (suited) / ``o`` (offsuit), pairs unsuffixed — e.g. ``"AKs"``,
``"AKo"``, ``"AA"``. The original table's uppercase ``S``/``O`` suffixes are
lowercased here to match the rest of the package.

Use :func:`top_range_hands` to build an opponent hand range (the top-X%
strongest starting hands), which the Monte Carlo engine samples opponents
from for range-aware equity.
"""

from __future__ import annotations

from typing import Set

# 169 hands. Values are win probabilities heads-up vs a random hand.
PREFLOP_EQUITIES = {
    "23o": 0.354, "24o": 0.362, "26o": 0.376, "34o": 0.379, "25o": 0.380,
    "27o": 0.381, "23s": 0.388, "36o": 0.390, "35o": 0.390, "37o": 0.396,
    "38o": 0.400, "24s": 0.402, "27s": 0.405, "28o": 0.406, "25s": 0.408,
    "26s": 0.410, "46o": 0.411, "47o": 0.417, "34s": 0.418, "45o": 0.418,
    "29o": 0.422, "48o": 0.423, "37s": 0.424, "35s": 0.425, "36s": 0.427,
    "39o": 0.427, "56o": 0.428, "28s": 0.429, "57o": 0.433, "49o": 0.435,
    "38s": 0.436, "2To": 0.436, "58o": 0.438, "45s": 0.443, "46s": 0.444,
    "47s": 0.445, "67o": 0.446, "3To": 0.453, "48s": 0.453, "56s": 0.454,
    "29s": 0.455, "39s": 0.457, "68o": 0.457, "59o": 0.459, "57s": 0.464,
    "4To": 0.465, "49s": 0.465, "2Jo": 0.466, "69o": 0.467, "5To": 0.467,
    "2Ts": 0.468, "78o": 0.472, "58s": 0.476, "6To": 0.477, "3Ts": 0.477,
    "79o": 0.479, "67s": 0.481, "3Jo": 0.481, "59s": 0.481, "68s": 0.483,
    "4Jo": 0.486, "2Qo": 0.493, "2Js": 0.494, "69s": 0.495, "3Qo": 0.497,
    "4Ts": 0.497, "6Jo": 0.497, "5Jo": 0.498, "5Ts": 0.501, "7To": 0.502,
    "89o": 0.504, "6Ts": 0.505, "78s": 0.506, "3Js": 0.512, "8To": 0.513,
    "7Jo": 0.515, "4Js": 0.517, "79s": 0.518, "7Ts": 0.519, "2Qs": 0.519,
    "22": 0.519, "4Qo": 0.521, "6Js": 0.522, "5Qo": 0.525, "5Js": 0.525,
    "2Ko": 0.528, "3Ko": 0.528, "89s": 0.529, "8Jo": 0.530, "9To": 0.530,
    "6Qo": 0.532, "4Qs": 0.533, "3Qs": 0.535, "7Qo": 0.536, "7Js": 0.539,
    "4Ko": 0.539, "8Ts": 0.539, "5Ko": 0.541, "9Jo": 0.547, "5Qs": 0.548,
    "2Ks": 0.551, "33": 0.556, "9Ts": 0.556, "8Qo": 0.557, "6Qs": 0.557,
    "8Js": 0.559, "7Qs": 0.560, "6Ko": 0.560, "4Ks": 0.564, "9Qo": 0.565,
    "3Ks": 0.565, "2Ao": 0.567, "8Ko": 0.567, "TJo": 0.568, "7Ko": 0.568,
    "3Ao": 0.574, "8Qs": 0.575, "5Ks": 0.576, "9Js": 0.580, "44": 0.582,
    "6Ks": 0.585, "TQo": 0.586, "2As": 0.588, "9Qs": 0.588, "7Ks": 0.590,
    "9Ko": 0.591, "JQo": 0.591, "4Ao": 0.592, "6Ao": 0.593, "5Ao": 0.594,
    "TJs": 0.594, "8Ks": 0.596, "7Ao": 0.600, "3As": 0.601, "TQs": 0.606,
    "TKo": 0.606, "4As": 0.607, "9Ao": 0.608, "JQs": 0.610, "JKo": 0.611,
    "9Ks": 0.615, "8Ao": 0.616, "55": 0.619, "6As": 0.620, "QKo": 0.624,
    "5As": 0.626, "7As": 0.628, "8As": 0.631, "TKs": 0.635, "TAo": 0.637,
    "66": 0.640, "QKs": 0.641, "9As": 0.643, "JKs": 0.644, "JAo": 0.646,
    "TAs": 0.650, "QAo": 0.651, "77": 0.659, "KAo": 0.659, "JAs": 0.661,
    "QAs": 0.667, "KAs": 0.682, "88": 0.698, "99": 0.720, "TT": 0.752,
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
