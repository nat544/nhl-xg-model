"""
Visualizations:
  1. A rink shot-location scatter, colored by predicted goal probability (xG).
  2. A cumulative xG-over-game-time line chart for a single game.
  3. A histogram of Monte Carlo simulated score margins, with the actual
     result marked, for one game.
"""

import matplotlib.pyplot as plt


def plot_shot_map(df, xg_col="xg", ax=None):
    """df needs columns: x, y, and an xg probability column."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    sc = ax.scatter(df["x"], df["y"], c=df[xg_col], cmap="Reds", s=25, alpha=0.8, edgecolors="k", linewidths=0.3)
    ax.set_xlim(-100, 100)
    ax.set_ylim(-45, 45)
    ax.set_title("Shot locations colored by predicted xG")
    ax.set_xlabel("x (ft from center ice)")
    ax.set_ylabel("y (ft from center line)")
    plt.colorbar(sc, ax=ax, label="xG")
    return ax


def plot_cumulative_xg(df_game, xg_col="xg", ax=None):
    """df_game: rows for ONE game, sorted by game_elapsed_s, with xg + is_home_team_shot."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4))

    df_sorted = df_game.sort_values("game_elapsed_s")
    for team_flag, label in [(True, "Home"), (False, "Away")]:
        team_df = df_sorted[df_sorted["is_home_team_shot"] == team_flag]
        ax.step(team_df["game_elapsed_s"] / 60, team_df[xg_col].cumsum(), where="post", label=label)

    ax.set_xlabel("Game time (minutes)")
    ax.set_ylabel("Cumulative xG")
    ax.set_title("Cumulative expected goals over game time")
    ax.legend()
    return ax


def plot_score_margin_distribution(sim_result, ax=None):
    """sim_result: output of simulate.simulate_game / actual_vs_simulated.
    Shows the spread of simulated outcomes and where the real result fell."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    margins = sim_result["score_margin_samples"]
    ax.hist(margins, bins=range(int(margins.min()), int(margins.max()) + 2), align="left", color="#c8102e", alpha=0.8)

    if "actual_margin" in sim_result:
        ax.axvline(sim_result["actual_margin"], color="black", linestyle="--", linewidth=2, label="Actual result")
        ax.legend()

    ax.set_xlabel(f"{sim_result['team_a']} goals minus {sim_result['team_b']} goals")
    ax.set_ylabel("Simulated games")
    ax.set_title("Monte Carlo simulated score margins")
    return ax
