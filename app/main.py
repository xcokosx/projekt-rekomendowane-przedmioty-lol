import numpy as np
from .train import train_model
from .predict import recommend_from_names
from .config import MODEL_PATH, FILE_PATH
from .neural_network import ItemRanker
from .mapping_loader import load_mappings

def train():
    print("Starting model training...")
    model = train_model()
    print("Training finished. Best model saved as best_model.npz")
    

def predict():
    # Load model
    model = ItemRanker.load(MODEL_PATH)

    # Load mappings
    champion_map, position_map, item_map = load_mappings()

    # Predict
    build = recommend_from_names(
        model,
        my_champion="Jinx",
        my_role="ADC",
        allies=["Thresh", "Lee Sin", "Ahri", "Garen"],
        enemies=["Lux", "Ezreal", "Kha'zix", "Teemo", "Zed"],
        item_map=item_map,
        champion_map=champion_map,
        position_map=position_map,
    )
    print("Recommended build:", build)

def print_mappings():
    champion_map, position_map, item_map = load_mappings()
    print("Champion Map:", champion_map)
    print("Position Map:", position_map)
    print("Item Map:", item_map)
    
if __name__ == "__main__":
    #print_mappings()
    predict()
    
