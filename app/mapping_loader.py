import numpy as np

def load_mappings(path="dataset/mappings.npz"):
    data = np.load(path, allow_pickle=True)
    return (
        data["champion_map"].item(),
        data["position_map"].item(),
        data["item_map"].item(),
    )
