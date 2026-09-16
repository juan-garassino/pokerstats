# CLAUDE.md — pokerStats package

`pokerStats` is an autonomous No-Limit Hold'em agent: it trains a CFR blueprint,
learns via PPO self-play with GTO distillation, encodes opponent tendencies, and
can play live via screen vision. Cards are two-char strings `"<rank><suit>"` with
rank in `"23456789TJQKA"` and suit in `"cdhs"` (lowercase, `treys`-compatible).

## Subpackages

- **`cfr/`** — MCCFR blueprint solver, hand abstraction, subgame solving, hybrid CFR+PPO agent.
- **`rl/`** — Gym environment (`PokerEnv`), PPO agent (`AlphaPokerNet`), self-play trainer, opponent encoder, MCTS, live bridge.
- **`vision/`** — screen reading (YOLO card detection + OCR) for live play.
- **`equity/`** — Monte Carlo poker equity engine (see below).

## `equity/` — Monte Carlo equity engine

Pure Python, stdlib only (no `treys`, no NumPy, no C++ extension). Ported and
rewritten from dickreuter/neuron_poker's `tools/` (MC equity + hand evaluator +
preflop table).

- **`cards.py`** — deck vocabulary, `make_deck(exclude)`, and `two_card_notation`
  (collapses two hole cards to a range key like `"AKs"`, `"AKo"`, `"AA"`).
- **`evaluator.py`** — dependency-free 5..7-card hand evaluator. `score_hand`
  returns a comparable `(category, kicker-tuple)`; `best_hand_index` resolves
  ties and returns all winning seats (for fractional split pots).
- **`preflop.py`** — 169-hand `PREFLOP_EQUITIES` table (heads-up vs random) and
  `top_range_hands(fraction)` for opponent-range modelling.
- **`montecarlo.py`** — `monte_carlo_equity(hole_cards, board, num_opponents,
  iterations, opponent_range, seed)` → `EquityResult(equity, win_rate, tie_rate,
  iterations)`. `opponent_range` is `1.0` (random), a smaller float (top-X% range),
  or an explicit set of notation keys. Split pots count fractionally.

```python
from pokerStats.equity import monte_carlo_equity
r = monte_carlo_equity(["As", "Ac"], num_opponents=1,
                       opponent_range={"KK"}, iterations=20_000, seed=7)
r.equity  # ~0.82  (AA vs KK preflop)
```

Tests: `tests/test_equity.py` (known equities: AA vs KK ≈ 0.82, AKs vs random ≈ 0.67,
made flush vs draw, multiway decay, range tightening).
