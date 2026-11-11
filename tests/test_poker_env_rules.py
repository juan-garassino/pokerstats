"""No-Limit Hold'em betting-rule tests for pokerStats.rl.poker_env.PokerEnv.

Motivated by the "Known issues" verified on 2026-11-11 (see CLAUDE.md):
  - heads-up the dealer posted the big blind and the SB acted first on every street;
  - `_advance()` closed a street as soon as bets matched, so the big blind never
    got its preflop option and the first postflop check ended the street.

Correct rules encoded here:
  - heads-up: dealer posts SB and acts first preflop; the BB acts first postflop;
  - 3+ players: SB = dealer+1, BB = dealer+2, preflop first = BB+1,
    postflop first = first seat after the dealer that can still act;
  - a street closes only when every player who can act has acted since the last
    bet/raise and all bets are matched (or at most one player can still act).

Seats are derived from `env.dealer_idx` after `reset()`, never hard-coded.
"""
from __future__ import annotations

from pokerStats.rl.poker_env import Action, PokerEnv, STREETS


def _env(num_players: int) -> PokerEnv:
    env = PokerEnv(num_players=num_players, small_blind=1.0, big_blind=2.0)
    env.reset()
    return env


def _street(env: PokerEnv) -> str:
    return STREETS[env.street_idx]


def _seats_3(env: PokerEnv) -> tuple[int, int, int]:
    d = env.dealer_idx
    return d, (d + 1) % 3, (d + 2) % 3  # btn, sb, bb


# ── Heads-up blinds and order ────────────────────────────────────────────────

def test_heads_up_dealer_posts_small_blind_and_acts_first_preflop():
    env = _env(2)
    dealer, other = env.dealer_idx, 1 - env.dealer_idx
    assert env.players[dealer].current_bet == 1.0
    assert env.players[other].current_bet == 2.0
    assert env.current_player == dealer


def test_heads_up_limp_gives_big_blind_option():
    env = _env(2)
    dealer, bb = env.dealer_idx, 1 - env.dealer_idx
    env.step(Action.CALL)  # dealer/SB limps
    assert _street(env) == "preflop"
    assert env.current_player == bb
    legal = env.legal_actions()
    assert Action.CHECK in legal
    assert Action.RAISE in legal


def test_heads_up_big_blind_acts_first_postflop():
    env = _env(2)
    dealer, bb = env.dealer_idx, 1 - env.dealer_idx
    env.step(Action.CALL)   # SB limps
    env.step(Action.CHECK)  # BB checks option
    assert _street(env) == "flop"
    assert env.current_player == bb


def test_heads_up_postflop_street_needs_two_checks():
    env = _env(2)
    dealer, bb = env.dealer_idx, 1 - env.dealer_idx
    env.step(Action.CALL)
    env.step(Action.CHECK)
    assert _street(env) == "flop"
    env.step(Action.CHECK)  # BB checks
    assert _street(env) == "flop"
    assert env.current_player == dealer
    env.step(Action.CHECK)  # dealer checks behind
    assert _street(env) == "turn"
    assert env.current_player == bb


# ── 3-player blinds, order and BB option ─────────────────────────────────────

def test_three_player_blinds_and_preflop_first_to_act():
    env = _env(3)
    btn, sb, bb = _seats_3(env)
    assert env.players[sb].current_bet == 1.0
    assert env.players[bb].current_bet == 2.0
    assert env.players[btn].current_bet == 0.0
    assert env.current_player == btn  # BB+1 wraps to the button 3-handed


def test_three_player_limp_around_gives_big_blind_option_to_check():
    env = _env(3)
    btn, sb, bb = _seats_3(env)
    env.step(Action.CALL)  # BTN limps
    env.step(Action.CALL)  # SB completes
    assert _street(env) == "preflop"
    assert env.current_player == bb
    legal = env.legal_actions()
    assert Action.CHECK in legal
    assert Action.RAISE in legal
    env.step(Action.CHECK)
    assert _street(env) == "flop"
    assert env.current_player == sb


def test_big_blind_option_raise_reopens_action():
    env = _env(3)
    btn, sb, bb = _seats_3(env)
    env.step(Action.CALL)
    env.step(Action.CALL)
    env.step(Action.RAISE, raise_frac=0.0)  # BB raises its option
    assert _street(env) == "preflop"
    assert env.current_player == btn
    env.step(Action.CALL)
    assert _street(env) == "preflop"
    assert env.current_player == sb
    env.step(Action.CALL)
    assert _street(env) == "flop"


# ── Postflop checks and re-opening ───────────────────────────────────────────

def _three_player_to_flop() -> PokerEnv:
    env = _env(3)
    env.step(Action.CALL)
    env.step(Action.CALL)
    env.step(Action.CHECK)
    assert _street(env) == "flop"
    return env


def test_three_player_postflop_street_needs_three_checks():
    env = _three_player_to_flop()
    btn, sb, bb = _seats_3(env)
    assert env.current_player == sb
    env.step(Action.CHECK)
    assert _street(env) == "flop" and env.current_player == bb
    env.step(Action.CHECK)
    assert _street(env) == "flop" and env.current_player == btn
    env.step(Action.CHECK)
    assert _street(env) == "turn" and env.current_player == sb


def test_three_player_check_down_reaches_showdown_after_every_check():
    env = _three_player_to_flop()
    steps = 0
    done = False
    while not done:
        _, rewards, done, info = env.step(Action.CHECK)
        steps += 1
        assert steps <= 9
    assert steps == 9  # 3 checks x flop, turn, river
    assert info["showdown"]
    assert abs(sum(rewards.values())) < 1e-6


def test_postflop_bet_after_check_reopens_action_for_checker():
    env = _three_player_to_flop()
    btn, sb, bb = _seats_3(env)
    env.step(Action.CHECK)                   # SB checks
    env.step(Action.RAISE, raise_frac=0.0)   # BB bets
    assert env.current_player == btn
    env.step(Action.CALL)                    # BTN calls
    assert _street(env) == "flop"
    assert env.current_player == sb          # checker must respond to the bet
    env.step(Action.CALL)
    assert _street(env) == "turn"


# ── Folds ─────────────────────────────────────────────────────────────────────

def test_fold_to_postflop_bet_ends_hand_heads_up():
    env = _env(2)
    dealer, bb = env.dealer_idx, 1 - env.dealer_idx
    env.step(Action.CALL)
    env.step(Action.CHECK)
    env.step(Action.RAISE, raise_frac=0.0)  # BB bets the flop
    _, rewards, done, info = env.step(Action.FOLD)
    assert done
    assert not info["showdown"]
    assert rewards[bb] > 0 > rewards[dealer]
    assert abs(sum(rewards.values())) < 1e-6


def test_fold_around_to_preflop_raise_ends_hand():
    env = _env(3)
    btn, sb, bb = _seats_3(env)
    env.step(Action.RAISE, raise_frac=0.0)
    env.step(Action.FOLD)
    _, rewards, done, _ = env.step(Action.FOLD)
    assert done
    assert rewards[btn] > 0
    assert abs(sum(rewards.values())) < 1e-6


# ── All-in handling ───────────────────────────────────────────────────────────

def test_all_in_player_is_skipped_on_later_streets():
    env = _env(3)
    btn, sb, bb = _seats_3(env)
    env.players[sb].stack = 9.0              # SB has 10 total behind incl. blind
    env.step(Action.CALL)                    # BTN limps
    env.step(Action.ALL_IN)                  # SB shoves to 10, reopening action
    assert env.players[sb].all_in
    assert env.current_player == bb
    env.step(Action.CALL)
    assert _street(env) == "preflop" and env.current_player == btn
    env.step(Action.CALL)
    assert _street(env) == "flop"
    assert env.current_player == bb          # SB is all-in and skipped
    env.step(Action.CHECK)
    assert env.current_player == btn
    env.step(Action.CHECK)
    assert _street(env) == "turn" and env.current_player == bb


def test_all_in_called_heads_up_runs_out_the_board():
    env = _env(2)
    env.step(Action.ALL_IN)                  # dealer shoves
    _, rewards, done, info = env.step(Action.CALL)
    assert done
    assert info["showdown"]
    assert len(env.community) == 5
    assert abs(sum(rewards.values())) < 1e-6


# ── Clone keeps per-street action state ───────────────────────────────────────

def test_clone_is_independent_of_original_action_state():
    env = _env(2)
    bb = 1 - env.dealer_idx
    env.step(Action.CALL)
    c = env.clone()
    c.step(Action.CHECK)
    assert _street(c) == "flop"
    assert _street(env) == "preflop" and env.current_player == bb
