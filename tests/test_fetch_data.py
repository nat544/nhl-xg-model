"""
Unit tests for fetch_data.py's parsing logic (extract_shot_events). These
use a small hand-built play-by-play fixture -- no network calls, no live API.
"""

from fetch_data import extract_shot_events

HOME_TEAM_ID = 10
AWAY_TEAM_ID = 20


def _fake_pbp(plays):
    return {
        "id": 999,
        "homeTeam": {"id": HOME_TEAM_ID},
        "awayTeam": {"id": AWAY_TEAM_ID},
        "plays": plays,
    }


def _shot_play(event_id, team_id, event_type="shot-on-goal", x=50, y=10):
    return {
        "eventId": event_id,
        "typeDescKey": event_type,
        "periodDescriptor": {"number": 1},
        "timeInPeriod": "01:00",
        "timeRemaining": "19:00",
        "situationCode": "1551",
        "homeTeamDefendingSide": "left",
        "details": {
            "eventOwnerTeamId": team_id,
            "xCoord": x,
            "yCoord": y,
            "zoneCode": "O",
            "shotType": "wrist",
            "shootingPlayerId": 111,
        },
    }


def _non_shot_play(event_type="faceoff"):
    return {"eventId": 1, "typeDescKey": event_type, "periodDescriptor": {"number": 1},
            "timeInPeriod": "00:00", "details": {}}


def test_non_shot_events_are_filtered_out():
    pbp = _fake_pbp([_non_shot_play("faceoff"), _non_shot_play("hit"), _non_shot_play("giveaway")])
    rows = extract_shot_events(pbp)
    assert rows == []


def test_extracts_all_four_shot_event_types():
    plays = [
        _shot_play(1, HOME_TEAM_ID, "shot-on-goal"),
        _shot_play(2, HOME_TEAM_ID, "goal"),
        _shot_play(3, HOME_TEAM_ID, "missed-shot"),
        _shot_play(4, HOME_TEAM_ID, "blocked-shot"),
    ]
    rows = extract_shot_events(_fake_pbp(plays))
    assert len(rows) == 4
    assert {r["event_type"] for r in rows} == {"shot-on-goal", "goal", "missed-shot", "blocked-shot"}


def test_home_and_away_shots_are_correctly_attributed():
    plays = [_shot_play(1, HOME_TEAM_ID), _shot_play(2, AWAY_TEAM_ID)]
    rows = extract_shot_events(_fake_pbp(plays))
    assert rows[0]["is_home_team_shot"] is True
    assert rows[1]["is_home_team_shot"] is False


def test_running_score_increments_only_on_goals_by_the_scoring_team():
    plays = [
        _shot_play(1, HOME_TEAM_ID, "goal"),           # home scores: 0-0 at time of shot, then 1-0
        _shot_play(2, AWAY_TEAM_ID, "shot-on-goal"),    # should see the updated home score
        _shot_play(3, AWAY_TEAM_ID, "goal"),            # away scores: 1-0 at time of shot, then 1-1
        _shot_play(4, HOME_TEAM_ID, "shot-on-goal"),    # should see the updated away score
    ]
    rows = extract_shot_events(_fake_pbp(plays))

    assert rows[0]["home_score_before"] == 0 and rows[0]["away_score_before"] == 0
    assert rows[1]["home_score_before"] == 1 and rows[1]["away_score_before"] == 0
    assert rows[2]["home_score_before"] == 1 and rows[2]["away_score_before"] == 0
    assert rows[3]["home_score_before"] == 1 and rows[3]["away_score_before"] == 1


def test_is_goal_flag_matches_event_type():
    plays = [_shot_play(1, HOME_TEAM_ID, "goal"), _shot_play(2, HOME_TEAM_ID, "shot-on-goal")]
    rows = extract_shot_events(_fake_pbp(plays))
    assert rows[0]["is_goal"] is True
    assert rows[1]["is_goal"] is False


def test_shooting_player_id_falls_back_to_scoring_player_id():
    play = _shot_play(1, HOME_TEAM_ID, "goal")
    del play["details"]["shootingPlayerId"]
    play["details"]["scoringPlayerId"] = 222
    rows = extract_shot_events(_fake_pbp([play]))
    assert rows[0]["shooting_player_id"] == 222


def test_extract_shot_events_on_empty_plays_returns_empty_list():
    assert extract_shot_events(_fake_pbp([])) == []
