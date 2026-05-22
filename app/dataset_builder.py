import pandas as pd
import numpy as np
from .data_preprocessor import build_mappings, DataPreprocessor
from .config import FILE_PATH, ITEM_COLS, EXCLUDED_ITEMS


def one_hot(index, size):
    one_hot = np.zeros(size, dtype=np.float32)
    if index >= 0:
        one_hot[index] = 1.0
    return one_hot

def build_team_vectors(df, row, champion_map):
    match_id = row["match_id"]
    team_id = row["team_id"]

    match_rows = df[df["match_id"] == match_id]

    ally_vec = np.zeros(len(champion_map), dtype=np.float32)
    enemy_vec = np.zeros(len(champion_map), dtype=np.float32)

    for _, p in match_rows.iterrows():
        champ_idx = p["champion_idx"]
        if champ_idx == -1:
            continue

        if p["team_id"] == team_id:
            ally_vec[champ_idx] += 1
        else:
            enemy_vec[champ_idx] += 1

    ally_vec /= 4
    enemy_vec /= 5

    return ally_vec, enemy_vec

def build_item_vector(row, item_map):
    item_vectors = []

    for col in ITEM_COLS:
        item_idx = row[col + "_idx"]
        item_vec = one_hot(item_idx, len(item_map))
        item_vectors.append(item_vec)
    
    return item_vectors

def normalize_gold(row):
    minutes = row["gameDuration"] / 60
    gpm = row["goldEarned"] / max(minutes, 1e-6)

    gold_norm = np.log1p(gpm) / np.log1p(1000)
    return min(1.0, gold_norm)

def build_input_vector(df, row, champion_map, item_map, position_map):
    champ_vec = one_hot(row["champion_idx"], size=len(champion_map))
    pos_vec = one_hot(row["position_idx"], size=len(position_map))

    ally_vec, enemy_vec = build_team_vectors(df, row, champion_map)
    item_vecs = build_item_vector(row, item_map)

    gold_vec = np.array([normalize_gold(row)], dtype=np.float32)
    
    input_vec = np.concatenate([champ_vec, pos_vec, ally_vec, enemy_vec] + item_vecs + [gold_vec])

    return input_vec

def build_dataset(df, champion_map, item_map, position_map):
    X = []
    y = []
    for _, row in df.iterrows():
        X.append(build_input_vector(df, row, champion_map, item_map, position_map))
        y.append(1 if row["win"] else 0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
