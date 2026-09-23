"""
Unit tests for simulate.py's Monte Carlo game simulation.
"""

import numpy as np
import pytest

from simulate import simulate_game, actual_vs_simulated


def test_zero_probability_shots_never_score():
    probs = [0.0, 0.0, 0.0]
    teams = ["A", "A", "B"]
    result = simulate_game(probs, teams, n_simulations=500, random_state=1)
    assert (result["team_a_goals_samples"] == 0).all()
    assert (result["team_b_goals_samples"] == 0).all()


def test_certain_shots_always_score():
    probs = [1.0, 1.0]
    teams = ["A", "B"]
    result = simulate_game(probs, teams, n_simulations=500, random_state=1)
    assert (result["team_a_goals_samples"] == 1).all()
    assert (result["team_b_goals_samples"] == 1).all()
    assert result["tie_prob"] == 1.0


def test_win_probabilities_sum_to_one():
    probs = [0.3, 0.4, 0.2, 0.6]
    teams = ["A", "B", "A", "B"]
    result = simulate_game(probs, teams, n_simulations=2000, random_state=1)
    total = result["team_a_win_prob"] + result["team_b_win_prob"] + result["tie_prob"]
    assert total == pytest.approx(1.0)


def test_mean_simulated_goals_converges_to_sum_of_probabilities():
    # Law of large numbers: with enough simulations, average goals should
    # approach the sum of per-shot probabilities for that team.
    team_a_probs = [0.5] * 20
    probs = team_a_probs + [0.1]  # team B needs at least one shot
    teams = ["A"] * 20 + ["B"]
    result = simulate_game(probs, teams, n_simulations=20000, random_state=1)
    expected_a = sum(team_a_probs)
    assert np.mean(result["team_a_goals_samples"]) == pytest.approx(expected_a, abs=0.1)


def test_raises_on_more_than_two_teams():
    with pytest.raises(ValueError):
        simulate_game([0.5, 0.5, 0.5], ["A", "B", "C"])


def test_raises_on_only_one_team():
    with pytest.raises(ValueError):
        simulate_game([0.5, 0.5], ["A", "A"])


def test_actual_vs_simulated_percentile_at_extremes():
    probs = [0.5, 0.5]
    teams = ["A", "B"]
    # An impossibly lopsided actual result should sit above ~all simulations
    result = actual_vs_simulated(probs, teams, actual_goals_a=100, actual_goals_b=0,
                                  n_simulations=2000, random_state=1)
    assert result["actual_percentile"] == pytest.approx(1.0)


def test_actual_vs_simulated_records_actual_margin():
    result = actual_vs_simulated([0.5, 0.5], ["A", "B"], actual_goals_a=3, actual_goals_b=1,
                                  n_simulations=100, random_state=1)
    assert result["actual_margin"] == 2
