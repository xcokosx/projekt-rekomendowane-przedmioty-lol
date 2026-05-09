import requests
import psycopg2
import json
import time

API_KEY = "RGAPI-5b4f33f2-a6e7-4b3b-ac2d-17d0acb05d15"

headers = {
    "X-Riot-Token": API_KEY
}

conn = psycopg2.connect(
    dbname="lol_ml",
    user="postgres",
    password="haslo",
    host="localhost",
    port="5432"
)

def riot_get(url):
    try:
        response = requests.get(url, headers=headers, timeout=10)

        print(f"GET {url} -> {response.status_code}")

        if response.status_code != 200:
            print("ERROR:", response.text)
            return None

        return response.json()

    except requests.RequestException as e:
        print("REQUEST FAILED:", e)
        return None
    
def load_players():
    with open("app/data/player_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["entries"]

def get_match_ids(puuid, count=50):
    url = (
        "https://europe.api.riotgames.com"
        f"/lol/match/v5/matches/by-puuid/{puuid}/ids"
        f"?start=0&count={count}"
    )
    return riot_get(url) or []

def get_match(match_id):
    url = (
        "https://europe.api.riotgames.com"
        f"/lol/match/v5/matches/{match_id}"
    )
    return riot_get(url)

cursor = conn.cursor()

def save_player(match_id, p):

    cursor.execute("""
        INSERT INTO match_players (
            match_id,
            player_puuid,
            team_id,
            win,
            champion,
            position,
            kills, deaths, assists,
            item0, item1, item2,
            item3, item4, item5, item6
        )
        VALUES (
            %s,%s,%s,%s,%s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s,%s,%s
        )
    """, (
        match_id,
        p.get("puuid"),

        0 if p.get("teamId") == 100 else 1,

        p.get("win"),

        p.get("championName"),
        p.get("teamPosition"),

        p.get("kills", 0),
        p.get("deaths", 0),
        p.get("assists", 0),

        p.get("item0", 0),
        p.get("item1", 0),
        p.get("item2", 0),
        p.get("item3", 0),
        p.get("item4", 0),
        p.get("item5", 0),
        p.get("item6", 0)
    ))

def save_match_ids(match_ids):

    with open("app/data/match_ids.json", "w", encoding="utf-8") as f:
        json.dump(match_ids, f, indent=4)

def run():

    print("START")

    players = load_players()

    unique_match_ids = set()


    # Zebranie match_id

    for player in players:

        puuid = player["puuid"]

        print(f"PLAYER: {puuid[:20]}...")

        try:
            match_ids = get_match_ids(puuid)

            unique_match_ids.update(match_ids)

        except Exception as e:
            print("MATCH ID ERROR:", e)

        time.sleep(1.2)

    unique_match_ids = list(unique_match_ids)

    print(f"Unique match_id: {len(unique_match_ids)}")

    save_match_ids(unique_match_ids)


    # Pobieranie danych z meczów

    for match_id in unique_match_ids:

        try:

            data = get_match(match_id)

            if not data:
                continue

            participants = data["info"]["participants"]

            print(f"{match_id} -> {len(participants)} players")

            for p in participants:
                save_player(match_id, p)

            conn.commit()

            print(f"SAVED {match_id}")

        except Exception as e:

            conn.rollback()

            print(f"MATCH ERROR {match_id}: {e}")

        time.sleep(1.2)

    print("DONE")


run()

cursor.close()
conn.close()