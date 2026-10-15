# PokerStats AI

No-Limit Hold'em research code: a Monte Carlo CFR blueprint solver, a PPO self-play
agent with an opponent-history encoder and optional CFR distillation, a Gym-style
poker environment, and a pure-Python equity engine.

**Status.** Work in progress. The environment, solvers, agent and tests run on CPU.
The repository contains no benchmark results or trained-strength evaluations, so none
are claimed here. Screen-reading code for live play (YOLO + OCR) exists only under
`reference/` and is not wired into the `pokerStats` package; `pokerStats/vision/` is an
empty placeholder.

## Contents

- [Quick start](#quick-start)
- [What is in the package](#what-is-in-the-package)
- [Network architecture](#network-architecture)
- [Training pipeline](#training-pipeline)
- [Make targets](#make-targets)
- [Data types and conventions](#data-types-and-conventions)
- [Outputs](#outputs)
- [Known issues](#known-issues)

## Quick start

Requires Python 3.12+ (`.python-version`). The tests and demos were run with only a
subset of the dependencies in `pyproject.toml`: `torch` (CPU build), `numpy`,
`gymnasium`, `treys`, `scipy`, `scikit-learn`, `pytest`.

```bash
uv venv && source .venv/bin/activate
uv pip install torch numpy gymnasium treys scipy scikit-learn pytest

python -m pytest tests -q          # 111 tests, about 2 minutes on CPU
PYTHONPATH=. python scripts/demo_play.py --hands 1 --delay 0   # random agent, 3 players
```

`uv sync` installs the full `pyproject.toml` list (including `mlflow`, `jupyterlab` and
cloud clients); it was not run for this README.

The scripts import `pokerStats` directly, so run them from the repo root with
`PYTHONPATH=.` or install the package first (`pip install .`). The `make demo*` targets
call `python scripts/...` without setting a path and assume the package is installed.

Equity example (from `pokerStats/CLAUDE.md`, output verified):

```python
from pokerStats.equity import monte_carlo_equity
r = monte_carlo_equity(["As", "Ac"], num_opponents=1,
                       opponent_range={"KK"}, iterations=20_000, seed=7)
r.equity   # 0.8155 with this seed (AA vs KK preflop)
```

## What is in the package

```
pokerStats/
├── rl/
│   ├── poker_env.py           # NLHE Gym environment, up to 9 seats (OBS_DIM = 160)
│   ├── ppo_agent.py           # PokerNet, AlphaPokerNet, RolloutBuffer, PPOAgent
│   ├── self_play_trainer.py   # 3-phase curriculum, League, ScriptedAgent archetypes
│   ├── opponent_encoder.py    # Transformer encoder over per-seat action history
│   ├── mcts.py                # PokerMCTS lookahead
│   ├── renderer.py            # Unicode terminal rendering
│   └── rl_live_bridge.py      # Checkpoint -> Decision for a screen-derived state
├── cfr/
│   ├── abstraction.py         # Hand bucketing (169 preflop + k-means postflop)
│   ├── cfr_solver.py          # External-sampling MCCFR (+ Kuhn poker test solver)
│   ├── blueprint.py           # Strategy table extraction, save/load (.npz)
│   ├── distillation.py        # BlueprintTeacher: CFR targets for PPO training
│   ├── hybrid_agent.py        # Blended CFR + PPO agent (inference)
│   ├── subgame_solver.py      # Real-time subgame refinement
│   └── cfr_live_bridge.py     # CFR decision engine for a screen-derived state
├── equity/                    # Monte Carlo equity, stdlib only (no treys/NumPy)
├── poker.py                   # Unrelated standalone 5-card ranking exercise
└── vision/                    # Placeholder
scripts/    demo_play.py, eval_rl.py, train_cfr.py
tests/      10 files, 111 tests
reference/  Read-only target architecture (not imported by the package)
notebooks/  Exploratory notebooks (train_rl, statistics, Untitled)
```

## Network architecture

`AlphaPokerNet` in `pokerStats/rl/ppo_agent.py`:

```
obs (160) + 8 opponent latents (8 x 32) -> 416
  Opponent encoder (Transformer, shared across seats)
  Trunk: Linear + LayerNorm, then 3 residual blocks, 256 hidden
  Heads
    actor        5 logits (fold / check / call / raise / all-in), legal-action masked
    raise        Beta(alpha, beta) for raise size in [0, 1]
    critic       scalar value
    cfr          7-way blueprint strategy prediction (distillation)
  Auxiliary (training only)
    card predictor (52 per opponent), action predictor (5 per opponent),
    range head (8 x 52)
```

`PokerNet` is the older network without opponent encoding, kept for old checkpoints.

## Training pipeline

1. **CFR (CPU).** Build the hand abstraction, run MCCFR, extract a blueprint
   (`scripts/train_cfr.py --phase abstraction|solve|blueprint`).
2. **PPO self-play (GPU recommended).** `MultiAgentTrainer` runs three phases;
   with a `BlueprintTeacher` it adds a distillation loss weighted by
   `cfr_coef_phase{1,2,3}` (defaults 1.0, 0.3, 0.1 in `CFG`).

| Phase | Opponents | Default hands in `CFG` | Makefile default |
|-------|-----------|------------------------|------------------|
| 1 | Self-play, all seats share weights | 30,000 | 20,000 |
| 2 | League of frozen snapshots (max 8, ELO-based) | 200,000 | 50,000 |
| 3 | Scripted archetypes (nit, fish, maniac, tight-aggressive) | 70,000 | 20,000 |

Size presets (`Makefile`; PPO hands are summed over the three phases):

| Size | CFR iterations | PPO hands |
|------|----------------|-----------|
| XXS | 100 | 2,000 |
| XS | 1,000 | 20,000 |
| S | 3,000 | 50,000 |
| M | 10,000 | 90,000 |
| L | 50,000 | 300,000 |
| XL | 250,000 | 500,000 |
| XXL | 500,000 | 700,000 |

The only timing figure in the repo is a Makefile comment: about 0.5 CFR iterations/s
on a Colab CPU. No other run times are recorded.

## Make targets

Run `make -n <target>` to see what a target does. Training targets need a GPU or a lot
of patience; `cfr-xxs` followed by `ppo-xxs` is the smallest end-to-end pipeline check.

| Group | Targets |
|-------|---------|
| Tests | `test` (needs `coverage`), `cfr-test`, `cfr-test-kuhn` |
| PPO only | `train`, `train-render`, `train-colab`, `train-colab-quick`, `train-colab-full`, `train-colab-{2,3,6,8}max` |
| CFR + PPO | `train-hybrid`, `train-unified` (distillation), and `-quick` / `-6max` / `-8max` variants |
| By size | `cfr-<size>`, `ppo-<size>`, `train-<size>` for xxs, xs, s, m, l, xl, xxl |
| Demos | `demo`, `demo-6max`, `demo-search`, `demo-vs-maniac`, `demo-vs-nit`, `demo-vs-tag`, `demo-trained`, `demo-trained-search` |
| Eval | `eval-rl`, `eval-rl-legacy` |
| Packaging | `zip-repo`, `zip-output`, `zip-cfr`, `zip-ppo` |

`demo-trained*` expect `checkpoints/best.pt`, which is not in the repo.
Variables: `NUM_PLAYERS` (6), `PHASE1_HANDS` / `PHASE2_HANDS` / `PHASE3_HANDS`,
`RENDER_EVERY` (5000), `USE_CFR` (0), `CFR_ITERS` (5000), `CFR_BUCKETS` (50),
`CFR_SAMPLES` (1000). Example:
`make train-colab NUM_PLAYERS=3 PHASE1_HANDS=10000 RENDER_EVERY=1000`.

Split CPU/GPU run on Colab: `make zip-repo`, then `make cfr-<size>` on a CPU instance,
`make zip-output` (or `zip-cfr`), and `make ppo-<size>` on a GPU instance.

## Data types and conventions

```python
class Action(IntEnum):
    FOLD = 0; CHECK = 1; CALL = 2
    RAISE = 3     # size from raise_frac in [0, 1]
    ALL_IN = 4
```

`raise_frac_to_amount` maps `[0, 1]` to `[min_raise, max_raise]` with
`(e^(3f) - 1) / (e^3 - 1)`.

Observation layout (`OBS_DIM = 160`, padded to 9 seats; empty seats are zero):

| Slice | Content |
|-------|---------|
| `[0:52]` | Hero hole cards (one-hot) |
| `[52:104]` | Community cards (one-hot) |
| `[104:113]` | Stacks, normalized (9 slots) |
| `[113:122]` | Current bets, normalized |
| `[122:131]` | Active flags |
| `[131:140]` | Positions relative to the dealer |
| `[140:144]` | Pot, call amount, min raise, active players |
| `[144:148]` | Street one-hot |
| `[148:155]` | Last 7 actions (id / `NUM_ACTIONS`) |
| `[155]`, `[156]` | Hand strength, draw potential |
| `[157:160]` | Raises this street / max, facing-raise flag, pot odds |

Cards are `"<rank><suit>"`, ranks `23456789TJQKA` (`T` = ten), suits `cdhs`, for example
`"Ah"`, `"Tc"`, `"2s"`. Money is float dollars.

## Outputs

Written to `checkpoints/` (listed in `.gitignore`):

| File | Description |
|------|-------------|
| `best.pt` | Best PPO checkpoint by ELO |
| `final.pt` | Final checkpoint |
| `ckpt_hand_XXXXXXXX.pt` | Periodic checkpoints |
| `cfr/blueprint.npz`, `cfr/abstraction.npz`, `cfr/solver_final.npz` | CFR artifacts |

## Known issues

- `pokerStats/rl/rl_live_bridge.py` builds observations with an older 145-dim slice
  layout that does not match `poker_env.py` (160 dims).
- `tests/test_renderer.py::test_render_table_output` fails (asserts `"Pot"`, renderer
  prints `pot`).
- `tests/test_hybrid_agent.py::test_blueprint_strategy_lookup` is occasionally flaky
  (unseeded solver, tight float tolerance).
- Running the tests overwrites the tracked `checkpoints/final.pt`.

## License

Private project.
