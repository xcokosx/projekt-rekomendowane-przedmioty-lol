import requests
import psycopg2

# ======================
# CONFIG
# ======================

API_KEY = "RGAPI-f761ee71-9b8f-4750-9ebe-8b50eed5a644"

MATCH_ID = "EUW1_7012345678" 

conn = psycopg2.connect(
    dbname="lol_ml",
    user="postgres",
    password="haslo",
    host="localhost",
    port="5432"
)

cursor = conn.cursor()


# ======================
# RIOT API FETCH
# ======================

def get_match(match_id):
    url = f"https://europe.api.riotgames.com/lol/match/v5/matches/{match_id}"
    headers = {"X-Riot-Token": API_KEY}

    response = requests.get(url, headers=headers)

    print("STATUS:", response.status_code)

    if response.status_code != 200:
        print("ERROR RESPONSE:", response.text)
        return None

    return response.json()


# ======================
# SAVE TO DB
# ======================

def save_player(match_id, p):
    try:
        cursor.execute("""
            INSERT INTO match_players (
                match_id, player_puuid, team_id, win,
                champion, position,
                kills, deaths, assists,
                item0, item1, item2, item3, item4, item5, item6
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
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

        conn.commit()
        print("✔ zapisano gracza")

    except Exception as e:
        print("DB ERROR:", e)


# ======================
# MAIN FLOW
# ======================

def run():
    print("START")

    data = get_match(MATCH_ID)

    if not data:
        print("Brak danych z API")
        return

    participants = data["info"]["participants"]

    print("Players:", len(participants))

    for p in participants:
        save_player(MATCH_ID, p)

    print("DONE")


run()