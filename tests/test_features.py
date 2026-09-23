"""
Unit tests for features.py -- the pure calculation logic (geometry, time
conversion, strength-state parsing) that doesn't need any network access.
"""

import pandas as pd
import pytest

from features import _time_to_seconds, _strength_state, build_features


# ---- _time_to_seconds -------------------------------------------------

def test_time_to_seconds_parses_mmss():
    assert _time_to_seconds("05:30") == 330


def test_time_to_seconds_handles_zero():
    assert _time_to_seconds("00:00") == 0


def test_time_to_seconds_handles_none():
    assert _time_to_seconds(None) is None


def test_time_to_seconds_handles_empty_string():
    assert _time_to_seconds("") is None


# ---- _strength_state ----------------------------------------------------

def test_strength_state_even_strength_home():
    # "1551": away has 5 skaters + goalie, home has 5 skaters + goalie
    assert _strength_state("1551", is_home_shot=True) == "5v5"


def test_strength_state_even_strength_away():
    assert _strength_state("1551", is_home_shot=False) == "5v5"


def test_strength_state_home_power_play():
    # home has 5 skaters, away down to 4 (short-handed)
    assert _strength_state("1451", is_home_shot=True) == "5v4"
    assert _strength_state("1451", is_home_shot=False) == "4v5"


def test_strength_state_handles_malformed_code():
    assert _strength_state("", is_home_shot=True) == "unknown"
    assert _strength_state(None, is_home_shot=True) == "unknown"
    assert _strength_state("55", is_home_shot=True) == "unknown"


# ---- build_features: geometry -------------------------------------------

def _base_row(**overrides):
    row = {
        "game_id": 1,
        "event_id": 1,
        "event_type": "shot-on-goal",
        "period": 1,
        "time_in_period": "00:00",
        "time_remaining": "20:00",
        "situation_code": "1551",
        "home_defending_side": "left",
        "x": 89,
        "y": 0,
        "zone": "O",
        "shot_type": "wrist",
        "shooting_player_id": 123,
        "owner_team_id": 10,
        "is_home_team_shot": True,
        "is_goal": False,
        "home_score_before": 0,
        "away_score_before": 0,
    }
    row.update(overrides)
    return row


def test_shot_right_at_the_net_has_near_zero_distance():
    df = build_features([_base_row(x=89, y=0)])
    assert df.iloc[0]["distance_ft"] == pytest.approx(0, abs=0.01)


def test_shot_from_center_ice_is_far_from_the_net():
    df = build_features([_base_row(x=0, y=0)])
    assert df.iloc[0]["distance_ft"] == pytest.approx(89, abs=0.01)


def test_distance_is_symmetric_regardless_of_attacking_direction():
    # A shot from the negative-x side, mirrored, should have the same
    # distance as the equivalent shot from the positive-x side.
    df_pos = build_features([_base_row(x=70, y=10)])
    df_neg = build_features([_base_row(x=-70, y=10)])
    assert df_pos.iloc[0]["distance_ft"] == pytest.approx(df_neg.iloc[0]["distance_ft"])


def test_shot_directly_in_front_of_net_has_zero_angle():
    # Same y as the net (0), directly out from it -> angle should be ~0 degrees
    df = build_features([_base_row(x=60, y=0)])
    assert df.iloc[0]["angle_deg"] == pytest.approx(0, abs=0.01)


# ---- build_features: game time -------------------------------------------

def test_game_elapsed_seconds_first_period():
    df = build_features([_base_row(period=1, time_in_period="05:00")])
    assert df.iloc[0]["game_elapsed_s"] == 300


def test_game_elapsed_seconds_second_period():
    # Period 2 starting point = 1 full period (1200s) already elapsed
    df = build_features([_base_row(period=2, time_in_period="00:00")])
    assert df.iloc[0]["game_elapsed_s"] == 1200


# ---- build_features: score differential -----------------------------------

def test_score_diff_from_home_shooter_perspective():
    df = build_features([_base_row(is_home_team_shot=True, home_score_before=3, away_score_before=1)])
    assert df.iloc[0]["score_diff"] == 2


def test_score_diff_from_away_shooter_perspective():
    df = build_features([_base_row(is_home_team_shot=False, home_score_before=3, away_score_before=1)])
    assert df.iloc[0]["score_diff"] == -2


# ---- build_features: target / empty input ----------------------------------

def test_target_is_1_for_goals_and_0_otherwise():
    df = build_features([_base_row(is_goal=True), _base_row(is_goal=False)])
    assert df["target"].tolist() == [1, 0]


def test_build_features_on_empty_input_returns_empty_dataframe():
    df = build_features([])
    assert df.empty


def test_missing_shot_type_becomes_unknown():
    df = build_features([_base_row(shot_type=None)])
    assert df.iloc[0]["shot_type"] == "unknown"
