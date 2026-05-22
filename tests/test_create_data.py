import pandas as pd
import app.create_data as cd


def test_flatten_participant():
    match = {
        "metadata": {"matchId": "MATCH123"},
        "info": {
            "mapId": 11,
            "gameDuration": 1800,
        }
    }

    p = {
        "puuid": "P1",
        "teamId": 100,
        "win": True,
        "goldEarned": 5000,
        "championName": "Jinx",
        "teamPosition": "BOTTOM",
        "kills": 5,
        "deaths": 2,
        "assists": 7,
        "item0": 1001,
        "item1": 2003,
        "item2": 0,
        "item3": 0,
        "item4": 0,
        "item5": 0,
        "item6": 0,
    }

    row = cd.flatten_participant(match, p)

    assert row["match_id"] == "MATCH123"
    assert row["champion"] == "Jinx"
    assert row["kills"] == 5
    assert row["item0"] == 1001
    assert row["team_id"] == 0


def test_dataframe_creation():
    rows = [
        {"match_id": "1", "champion": "Jinx", "kills": 5},
        {"match_id": "2", "champion": "Lux", "kills": 3},
    ]

    df = pd.DataFrame(rows)

    assert len(df) == 2
    assert list(df.columns) == ["match_id", "champion", "kills"]
