""" Features: shot distance/angle to the net, time elapsed in the game, and the
score differential at the moment of the shot.
"""

import math
import pandas as pd

NET_X = 89.0  # NHL rink: net sits ~89ft from center ice along the long axis


def _time_to_seconds(mmss: str) -> int:
    if not mmss:
        return None
    m, s = mmss.split(":")
    return int(m) * 60 + int(s)


def _strength_state(situation_code: str, is_home_shot: bool) -> str:
    """
    situationCode is 4 digits: [away_goalie][away_skaters][home_skaters][home_goalie]
    e.g. "1551" = away has goalie(1) + 5 skaters, home has 5 skaters + goalie(1) -> even strength.
    Returns something like "5v5", "5v4", "4v5", etc. from the shooting team's perspective.
    """
    if not situation_code or len(situation_code) != 4:
        return "unknown"
    away_skaters, home_skaters = int(situation_code[1]), int(situation_code[2])
    shooter_skaters, opp_skaters = (home_skaters, away_skaters) if is_home_shot else (away_skaters, home_skaters)
    return f"{shooter_skaters}v{opp_skaters}"


def build_features(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Normalize x so "distance to net" always measures toward the attacking net,
    # regardless of which side of the rink the team is shooting from.
    def attacking_net_x(row):
        return NET_X if row["x"] >= 0 else -NET_X

    df["net_x"] = df.apply(attacking_net_x, axis=1)
    df["distance_ft"] = ((df["net_x"] - df["x"]) ** 2 + df["y"] ** 2) ** 0.5
    df["angle_deg"] = df.apply(
        lambda r: math.degrees(math.atan2(abs(r["y"]), abs(r["net_x"] - r["x"]) + 1e-6)),
        axis=1,
    )

    df["time_in_period_s"] = df["time_in_period"].map(_time_to_seconds)
    df["game_elapsed_s"] = (df["period"] - 1) * 1200 + df["time_in_period_s"]

    df["score_diff"] = df.apply(
        lambda r: (r["home_score_before"] - r["away_score_before"]) if r["is_home_team_shot"]
        else (r["away_score_before"] - r["home_score_before"]),
        axis=1,
    )

    df["strength_state"] = df.apply(
        lambda r: _strength_state(r["situation_code"], r["is_home_team_shot"]), axis=1
    )

    df["shot_type"] = df["shot_type"].fillna("unknown")
    df["target"] = df["is_goal"].astype(int)

    feature_cols = [
        "distance_ft", "angle_deg", "game_elapsed_s", "score_diff",
        "period", "shot_type", "strength_state",
    ]
    return df[feature_cols + ["target", "game_id", "event_id", "x", "y", "is_home_team_shot"]]
