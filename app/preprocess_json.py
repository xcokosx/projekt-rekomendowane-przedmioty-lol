import json

ROLE_ORDER = {
    "TOP": 0,
    "JUNGLE": 1,
    "MID": 2,
    "BOT": 3,
    "SUPPORT": 4
}

ROLE_MAP = {
    "TOP": "TOP",
    "JUNGLE": "JUNGLE",
    "MIDDLE": "MID",
    "BOTTOM": "BOT",
    "UTILITY": "SUPPORT"
}


def get_team_players(match_data, team_id):
    return [
        p for p in match_data
        if p.get("team_id") == team_id
    ]

def sort_players_by_role(players):

    def key(x):
        pos = x.get("position")

        mapped = ROLE_MAP.get(pos)

        if mapped is None:
            return 999

        return ROLE_ORDER[mapped]

    return sorted(players, key=key)


def extract_items(player):

    items = []

    for i in range(6):

        item = player[f"item{i}"]

        if item != 0:
            items.append(item)

    return items


def create_training_samples(match_data):

    blue_team = sort_players_by_role(
        get_team_players(match_data, 0)
    )

    red_team = sort_players_by_role(
        get_team_players(match_data, 1)
    )

    samples = []

    for player in match_data:

        current_team = player["team_id"]

        if current_team == 0:
            allies = blue_team
            enemies = red_team
        else:
            allies = red_team
            enemies = blue_team

        ally_champs = []

        for ally in allies:

            if ally["player_puuid"] != player["player_puuid"]:

                ally_champs.append(
                    ally["champion"]
                )

        enemy_champs = [
            enemy["champion"]
            for enemy in enemies
        ]

        sample = {

            "champion":
                player["champion"],

            "role":
                ROLE_MAP[player["position"]],

            "ally_team":
                ally_champs,

            "enemy_team":
                enemy_champs,

            "items":
                extract_items(player),

            "win":
                int(player["win"])
        }

        samples.append(sample)

    return samples


def clean_match(match):
    valid_positions = {"TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"}
    for p in match:
        assert_flat_player(p)
    return [
        p for p in match
        if (
            p.get("team_id") in (0, 1)
            and p.get("position") in valid_positions
            and isinstance(p.get("champion"), str)
            and isinstance(p.get("player_puuid"), str)
        )
    ]

def assert_flat_player(p):
    assert isinstance(p, dict)
    assert isinstance(p.get("champion"), str)
    assert isinstance(p.get("team_id"), int)
    assert "ally_team" not in p
    assert "enemy_team" not in p
    

with open("./app/matches.json", "r", encoding="utf-8") as f:
#with open("./app/match.json", "r", encoding="utf-8") as f:                 #DEBUGGING PURPOSES, SMALLER DATASET
    data = json.load(f)

from collections import defaultdict

# Group players by match_id so we process match-by-match (10 players per match)
by_match = defaultdict(list)
for p in data["match_players"]:
    mid = p.get("match_id")
    if mid is None:
        continue
    by_match[mid].append(p)

samples = []
for mid, players in by_match.items():
    # only full matches
    if len(players) != 10:
        continue

    try:
        match = clean_match(players)
    except AssertionError:
        # skip malformed match entries
        continue

    if len(match) != 10:
        continue

    samples.extend(create_training_samples(match))
#print(json.dumps(samples, indent=4)) ------------------- DO NOT UNCOMMENT, TOO MUCH DATA OVER 10000 RECORDS

output_path = "./app/preprocessed_match_champion_data.json"
#output_path = "./app/preprocessed_match_champion_data_testing.json"        #DEBUGGING PURPOSES, SMALLER DATASET
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=4, ensure_ascii=False)

print(f"Zapisano {len(samples)} rekordów do {output_path}")

"""
Input:

Json matches.json

Output:

{
  "champion": "Mel",
  "role": "MID",
  "ally_team": ["Yone", "Briar", "Twitch", "Karma"],
  "enemy_team": ["Renekton", "Lee Sin", "Ahri", "Veigar", "Lulu"],
  "items": [
    "Rabadon's Deathcap",
    "Luden's Echo",
    "Spellslinger's Shoes",
    "Shadowflame"
  ],
  "win": 0
}
.
.
.
{
    "champion": "Lulu",
    "role": "SUPPORT",
    "ally_team": ["Renekton", "Lee Sin", "Ahri", "Veigar"],
    "enemy_team": ["Yone", "Briar", "Mel", "Twitch", "Karma"],
    "items": ["Dream Maker", "Redemption", "Locket of the Iron Solari", "Ionian Boots of Lucidity"],
    "win": 1
}

"""