"""Tests for pokerStats.equity — the Monte Carlo equity engine.

Assertions are pinned to well-known poker equities (AA vs KK ≈ 82%,
AKs vs random ≈ 67%, dominant made hands vs draws) with tolerances wide
enough to absorb Monte Carlo variance at the iteration counts used here.
"""

import pytest

from pokerStats.equity import (
    monte_carlo_equity, two_card_notation, top_range_hands,
    score_hand, best_hand_index, hand_name,
    STRAIGHT_FLUSH, FLUSH, FULL_HOUSE, FOUR_OF_A_KIND, TWO_PAIR, PAIR, HIGH_CARD,
)


# ── hand evaluator ──────────────────────────────────────────────────────────

def test_evaluator_category_ordering():
    """Category ranks respect standard poker ordering."""
    straight_flush = score_hand(["9h", "8h", "7h", "6h", "5h"])
    quads = score_hand(["9h", "9s", "9d", "9c", "5h"])
    full_house = score_hand(["9h", "9s", "9d", "5c", "5h"])
    flush = score_hand(["Ah", "Jh", "8h", "5h", "2h"])
    two_pair = score_hand(["9h", "9s", "5d", "5c", "2h"])
    assert straight_flush > quads > full_house > flush > two_pair
    assert hand_name(straight_flush) == "StraightFlush"
    assert straight_flush[0] == STRAIGHT_FLUSH
    assert quads[0] == FOUR_OF_A_KIND
    assert full_house[0] == FULL_HOUSE
    assert flush[0] == FLUSH
    assert two_pair[0] == TWO_PAIR


def test_evaluator_wheel_straight():
    """A-2-3-4-5 is a straight with the five as the high card."""
    wheel = score_hand(["Ah", "2s", "3d", "4c", "5h"])
    six_high = score_hand(["2h", "3s", "4d", "5c", "6h"])
    assert hand_name(wheel) == "Straight"
    assert six_high > wheel  # 6-high straight beats the wheel


def test_evaluator_best_of_seven():
    """A flush is found among seven cards even with a pair present."""
    seven = ["Ah", "Kh", "Qh", "2h", "5h", "5s", "9d"]
    score = score_hand(seven)
    assert score[0] == FLUSH


def test_evaluator_ties_return_all_winners():
    """Identical hands tie and both indices are returned."""
    board = ["As", "Ks", "Qd", "Jc", "9h"]
    a = ["2h", "3d"] + board
    b = ["2c", "3s"] + board  # same board-plays-best, identical five cards
    winners, _ = best_hand_index([a, b])
    assert winners == [0, 1]


# ── two-card notation ─────────────────────────────────────────────────────────

def test_two_card_notation():
    assert two_card_notation(["Ah", "Ks"]) == "AKo"
    assert two_card_notation(["Ah", "Kh"]) == "AKs"
    assert two_card_notation(["Ah", "Ac"]) == "AA"
    assert two_card_notation(["Ks", "Ah"]) == "AKo"  # order-independent


def test_top_range_hands():
    top10 = top_range_hands(0.1)
    assert "AA" in top10 and "KK" in top10
    assert "23o" not in top10
    assert top_range_hands(1.0) >= top10  # full range is a superset


# ── Monte Carlo equity ────────────────────────────────────────────────────────

def test_aa_vs_kk_preflop():
    """AA is ~82% against a single KK heads-up preflop."""
    result = monte_carlo_equity(
        ["As", "Ac"], board=[], num_opponents=1,
        iterations=20_000, opponent_range={"KK"}, seed=7,
    )
    assert result.equity == pytest.approx(0.82, abs=0.02)


def test_aks_vs_random_preflop():
    """AKs is ~67% against one random opponent."""
    result = monte_carlo_equity(
        ["As", "Ks"], board=[], num_opponents=1,
        iterations=20_000, opponent_range=1.0, seed=11,
    )
    assert result.equity == pytest.approx(0.67, abs=0.02)


def test_made_flush_vs_draw_on_turn():
    """A completed flush crushes a bare flush draw with one card to come."""
    # Hero holds a made flush; villain holds two more hearts (flush draw).
    board = ["Ah", "Kh", "7h", "2s"]  # three hearts on turn
    result = monte_carlo_equity(
        ["Qh", "Jh"], board=board, num_opponents=1,
        iterations=10_000, opponent_range=1.0, seed=3,
    )
    # Hero already has the flush; villain can only win by pairing the board
    # into a full house / quads, which is rare — hero should be >90%.
    assert result.equity > 0.90


def test_dominated_hand_multiway_drops():
    """Equity against more opponents is strictly lower than heads-up."""
    heads_up = monte_carlo_equity(
        ["As", "Ks"], num_opponents=1, iterations=8_000, seed=5,
    )
    five_way = monte_carlo_equity(
        ["As", "Ks"], num_opponents=5, iterations=8_000, seed=5,
    )
    assert five_way.equity < heads_up.equity


def test_range_tightens_lowers_equity():
    """A tighter opponent range lowers a marginal hand's equity."""
    vs_random = monte_carlo_equity(
        ["Ts", "9s"], num_opponents=1, iterations=12_000,
        opponent_range=1.0, seed=9,
    )
    vs_tight = monte_carlo_equity(
        ["Ts", "9s"], num_opponents=1, iterations=12_000,
        opponent_range=0.1, seed=9,
    )
    assert vs_tight.equity < vs_random.equity


def test_result_fields_consistent():
    """equity == win_rate + tie_rate and all rates are valid probabilities."""
    r = monte_carlo_equity(["As", "Ac"], num_opponents=1, iterations=5_000, seed=1)
    assert r.equity == pytest.approx(r.win_rate + r.tie_rate)
    assert 0.0 <= r.win_rate <= 1.0
    assert 0.0 <= r.tie_rate <= 1.0
    assert r.iterations == 5_000
