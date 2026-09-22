"""
Pulls play-by-play data from the NHL's public API and extracts shot events
(shot-on-goal, goal, missed-shot, blocked-shot) into a flat list of dicts.

No API key needed. Endpoints used:
  - schedule:      https://api-web.nhle.com/v1/schedule/{YYYY-MM-DD}
  - play-by-play:  https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play

Schema verified directly against the live API. If the NHL changes their API
shape, tweak the field names in `extract_shot_events`.
"""

import time
import requests

BASE_URL = "https://api-web.nhle.com/v1"
SHOT_EVENT_TYPES = {"shot-on-goal", "goal", "missed-shot", "blocked-shot"}


def get_game_ids_for_date(date_str: str) -> list[int]:
    """date_str format: 'YYYY-MM-DD'. Returns NHL game IDs played that day."""
    resp = requests.get(f"{BASE_URL}/schedule/{date_str}", timeout=10)
    resp.raise_for_status()
    data = resp.json()
    game_ids = []
    for week in data.get("gameWeek", []):
        if week.get("date") == date_str:
            game_ids.extend(g["id"] for g in week.get("games", []))
    return game_ids


def get_play_by_play(game_id: int) -> dict:
    resp = requests.get(f"{BASE_URL}/gamecenter/{game_id}/play-by-play", timeout=10)
    resp.raise_for_status()
    return resp.json()


def extract_shot_events(pbp: dict) -> list[dict]:
    """
    Flattens the play-by-play JSON into one row per shot attempt, tracking a
    running score differential as we go (needed for the "score state" feature).
    """
    game_id = pbp.get("id")
    home_team_id = pbp.get("homeTeam", {}).get("id")
    away_team_id = pbp.get("awayTeam", {}).get("id")

    rows = []
    home_score = 0
    away_score = 0

    for play in pbp.get("plays", []):
        event_type = play.get("typeDescKey")

        if event_type not in SHOT_EVENT_TYPES:
            continue

        details = play.get("details", {})
        owner_team_id = details.get("eventOwnerTeamId")
        is_home_team_shot = owner_team_id == home_team_id

        rows.append({
            "game_id": game_id,
            "event_id": play.get("eventId"),
            "event_type": event_type,
            "period": play.get("periodDescriptor", {}).get("number"),
            "time_in_period": play.get("timeInPeriod"),      # "MM:SS"
            "time_remaining": play.get("timeRemaining"),     # "MM:SS"
            "situation_code": play.get("situationCode"),     # e.g. "1551" -> strength state
            "home_defending_side": play.get("homeTeamDefendingSide"),
            "x": details.get("xCoord"),
            "y": details.get("yCoord"),
            "zone": details.get("zoneCode"),
            "shot_type": details.get("shotType"),
            "shooting_player_id": details.get("shootingPlayerId") or details.get("scoringPlayerId"),
            "owner_team_id": owner_team_id,
            "is_home_team_shot": is_home_team_shot,
            "is_goal": event_type == "goal",
            "home_score_before": home_score,
            "away_score_before": away_score,
        })

        if event_type == "goal":
            if owner_team_id == home_team_id:
                home_score += 1
            elif owner_team_id == away_team_id:
                away_score += 1

    return rows


def fetch_shots_for_date_range(dates: list[str], sleep_sec: float = 0.3) -> list[dict]:
    """Convenience wrapper: given a list of 'YYYY-MM-DD' strings, pull every
    shot event from every game played on those dates. Be polite to the API —
    this sleeps briefly between game requests."""
    all_rows = []
    for date_str in dates:
        game_ids = get_game_ids_for_date(date_str)
        for gid in game_ids:
            pbp = get_play_by_play(gid)
            all_rows.extend(extract_shot_events(pbp))
            time.sleep(sleep_sec)
    return all_rows


if __name__ == "__main__":
    # Quick smoke test on a single known game.
    pbp = get_play_by_play(2023020672)
    shots = extract_shot_events(pbp)
    print(f"Extracted {len(shots)} shot events from game {pbp.get('id')}")
    print(shots[0])
