"""
Monte Carlo simulation of game outcomes from per-shot goal probabilities.

Each shot is treated as an independent Bernoulli trial with probability
equal to its predicted xG. Re-drawing every shot thousands of times gives a
distribution of possible final scores for the game that actually happened,
not just the single result that was observed. This separates "who won" from
"who generated the better scoring chances."
"""

import numpy as np
import pandas as pd


def simulate_game(shot_probs, team_labels, n_simulations=10000, random_state=None):
    """
    shot_probs: per-shot goal probability (xG), in the order shots occurred.
    team_labels: same length, identifies which team took each shot (exactly
        two distinct values expected, e.g. True/False for home/away).
    n_simulations: number of independent random replays.

    Shots are simulated independently (no momentum/rebound effects between
    draws) since that matches how the xG model itself was trained -- each
    shot's probability is estimated on its own, not conditioned on prior
    shots in the same game.

    Returns dict with win probabilities and the full sampled distributions
    of goals/margin, so callers can plot or do further analysis.
    """
    rng = np.random.default_rng(random_state)
    shot_probs = np.asarray(shot_probs, dtype=float)
    team_labels = np.asarray(team_labels)

    teams = pd.unique(team_labels)
    if len(teams) != 2:
        raise ValueError("simulate_game expects shots from exactly two teams")
    team_a, team_b = teams[0], teams[1]
    is_a = team_labels == team_a
    is_b = team_labels == team_b

    # (n_simulations x n_shots) matrix of independent Bernoulli(p) draws
    draws = rng.random((n_simulations, len(shot_probs))) < shot_probs

    goals_a = draws[:, is_a].sum(axis=1)
    goals_b = draws[:, is_b].sum(axis=1)
    margin = goals_a - goals_b

    return {
        "team_a": team_a,
        "team_b": team_b,
        "team_a_goals_samples": goals_a,
        "team_b_goals_samples": goals_b,
        "score_margin_samples": margin,
        "team_a_win_prob": float((margin > 0).mean()),
        "team_b_win_prob": float((margin < 0).mean()),
        "tie_prob": float((margin == 0).mean()),
    }


def actual_vs_simulated(shot_probs, team_labels, actual_goals_a, actual_goals_b,
                         n_simulations=10000, random_state=None):
    """
    Same as simulate_game, plus a comparison to what actually happened:
    actual_percentile is where the real score margin falls in the simulated
    distribution (e.g. 0.90 means the real result was more lopsided in team
    A's favor than 90% of simulated replays -- a rough "was this expected or
    lucky" check).
    """
    result = simulate_game(shot_probs, team_labels, n_simulations, random_state)
    actual_margin = actual_goals_a - actual_goals_b
    result["actual_margin"] = actual_margin
    result["actual_percentile"] = float((result["score_margin_samples"] <= actual_margin).mean())
    return result
