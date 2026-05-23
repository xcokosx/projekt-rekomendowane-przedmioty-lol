import numpy as np
import json
from .neural_network import ItemRanker
from .dataset_builder import one_hot


def load_model(model_path):
    """Load trained model using ItemRanker.load()."""
    return ItemRanker.load(model_path)

# Nazwy itemów zamiast id
def load_item_id_to_name_map(items_json_path="app/data/items.json"):
    """Load mapping from item_map index to item name using items.json."""
    
    with open(items_json_path, 'r') as f:
        items_data = json.load(f)
    
    id_to_name = {}
    for item in items_data.get('full_list', []):
        id_to_name[item['id']] = item['name']
   
    from .mapping_loader import load_mappings
    champion_map, position_map, item_map = load_mappings()

    idx_to_name = {}
    for item_id_str, idx in item_map.items():
        item_id = int(item_id_str)
        if item_id in id_to_name:
            idx_to_name[idx] = id_to_name[item_id]
        else:
            idx_to_name[idx] = f"Unknown(ID:{item_id})"
    
    return idx_to_name


def build_team_vectors_from_names(allies, enemies, champion_map):
    num_champs = len(champion_map)

    ally_vec = np.zeros(num_champs, dtype=np.float32)
    enemy_vec = np.zeros(num_champs, dtype=np.float32)

    for name in allies:
        if name in champion_map:
            ally_vec[champion_map[name]] = 1.0

    for name in enemies:
        if name in champion_map:
            enemy_vec[champion_map[name]] = 1.0

    return ally_vec, enemy_vec


def recommend_from_names(
    model,
    my_champion,
    my_role,
    allies,
    enemies,
    item_map,
    champion_map,
    position_map,
):
    
    champion_idx = champion_map[my_champion]
    pos_idx = position_map[my_role]

    ally_vec, enemy_vec = build_team_vectors_from_names(allies, enemies, champion_map)

    num_items = len(item_map)
    current_items = []
    build = []

    # neutral mid‑game gold value
    gold_norm = 0.5

    for _ in range(6):
        champ_vec = one_hot(champion_idx, size=len(champion_map))
        pos_vec = one_hot(pos_idx, size=len(position_map))

        item_state = np.zeros(num_items, dtype=np.float32)
        for ci in current_items:
            item_state[ci] = 1.0

        x_vec = np.concatenate([
            champ_vec,
            pos_vec,
            ally_vec,
            enemy_vec,
            item_state,
            np.array([gold_norm], dtype=np.float32),
        ])

        pred = model.forward(x_vec)

        # mask already purchased items
        for ci in current_items:
            pred[ci] = -1

        next_item = int(np.argmax(pred))
        current_items.append(next_item)
        build.append(next_item)

    return build
