CREATE TABLE match_players (
    id SERIAL PRIMARY KEY,

    match_id TEXT,
    player_puuid TEXT,

    team_id SMALLINT,
    win BOOLEAN,

    champion TEXT,
    position TEXT,

    kills INT,
    deaths INT,
    assists INT,

    item0 INT,
    item1 INT,
    item2 INT,
    item3 INT,
    item4 INT,
    item5 INT,
    item6 INT
);