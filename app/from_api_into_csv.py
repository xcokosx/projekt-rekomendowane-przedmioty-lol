import asyncio
import aiohttp
import pandas as pd
from aiolimiter import AsyncLimiter

API_KEY = "RGAPI-b01019ef-af32-4ac8-a56d-5840ca6bba5d"

HEADERS = {"X-Riot-Token": API_KEY}

REQ_PER_SEC = AsyncLimiter(20, 1)
REQ_PER_2MIN = AsyncLimiter(100, 120)

SEM = asyncio.Semaphore(3)
BATCH_SIZE = 3               

pause_lock = asyncio.Lock()

async def fetch(session, url):

    async with SEM:

        async with REQ_PER_SEC:
            async with REQ_PER_2MIN:

                async with session.get(url, headers=HEADERS) as r:

                    if r.status == 429:

                        async with pause_lock:

                            retry_after = r.headers.get("Retry-After")
                            wait_time = int(retry_after) if retry_after else 10

                            print(f"429 -> COOLDOWN {wait_time}s")

                            await asyncio.sleep(wait_time)

                        return None

                    if r.status != 200:
                        text = await r.text()
                        print("ERROR:", r.status, text)
                        return None

                    return await r.json()

async def get_challengers(session):

    url = (
        "https://eun1.api.riotgames.com/"
        "lol/league/v4/challengerleagues/"
        "by-queue/RANKED_SOLO_5x5"
    )

    data = await fetch(session, url)

    if not data:
        return []

    return [e["puuid"] for e in data["entries"]]

async def get_match_ids(session, puuid):

    url = (
        "https://europe.api.riotgames.com"
        f"/lol/match/v5/matches/by-puuid/{puuid}/ids"
        "?start=0&count=50"
    )

    return await fetch(session, url) or []


async def collect_match_ids(session, puuids):

    all_ids = set()

    for i in range(0, len(puuids), BATCH_SIZE):

        batch = puuids[i:i + BATCH_SIZE]

        tasks = [get_match_ids(session, p) for p in batch]

        results = await asyncio.gather(*tasks)

        for r in results:
            all_ids.update(r)

        print(f"Match ID batch {i}-{i+BATCH_SIZE}")

        await asyncio.sleep(1)  

    return list(all_ids)

async def get_match(session, match_id):

    url = (
        "https://europe.api.riotgames.com"
        f"/lol/match/v5/matches/{match_id}"
    )

    return await fetch(session, url)


async def collect_matches(session, match_ids):

    all_rows = []

    for i in range(0, len(match_ids), BATCH_SIZE):

        batch = match_ids[i:i + BATCH_SIZE]

        tasks = [get_match(session, m) for m in batch]

        results = await asyncio.gather(*tasks)

        for match in results:

            if not match:
                continue

            for p in match["info"]["participants"]:

                all_rows.append({
                    "match_id": match["metadata"]["matchId"],
                    "puuid": p.get("puuid"),

                    "team_id": 0 if p.get("teamId") == 100 else 1,
                    "win": p.get("win"),

                    "champion": p.get("championName"),
                    "position": p.get("teamPosition"),

                    "kills": p.get("kills", 0),
                    "deaths": p.get("deaths", 0),
                    "assists": p.get("assists", 0),

                    "item0": p.get("item0", 0),
                    "item1": p.get("item1", 0),
                    "item2": p.get("item2", 0),
                    "item3": p.get("item3", 0),
                    "item4": p.get("item4", 0),
                    "item5": p.get("item5", 0),
                    "item6": p.get("item6", 0),
                })

        print(f"Match batch {i}-{i+BATCH_SIZE}")

        await asyncio.sleep(1)

    return all_rows

async def main():

    async with aiohttp.ClientSession() as session:

        print("STEP 1: challengers")

        puuids = await get_challengers(session)

        print("Players:", len(puuids))

        print("STEP 2: match ids")

        match_ids = await collect_match_ids(session, puuids)

        print("Matches:", len(match_ids))

        print("STEP 3: match details")

        data = await collect_matches(session, match_ids)

        print("STEP 4: dataframe")

        df = pd.DataFrame(data)

        print(df.head())

        print("STEP 5: parquet save")

        df.to_parquet("matches.parquet", index=False)

        print("DONE")


if __name__ == "__main__":
    asyncio.run(main())