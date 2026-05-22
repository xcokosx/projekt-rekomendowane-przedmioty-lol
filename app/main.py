import numpy as np
from .train import train_model
from .predict import recommend_from_names
from .config import MODEL_PATH, FILE_PATH
from .neural_network import ItemRanker
from .mapping_loader import load_mappings
from .create_data import run
def train():
    print("Starting model training...")
    model = train_model()
    print("Training finished. Best model saved as best_model.npz")
    

def predict(my_champion="Jinx",
        my_role="ADC",
        allies=["Thresh", "Lee Sin", "Ahri", "Garen"],
        enemies=["Lux", "Ezreal", "Kha'zix", "Teemo", "Zed"]):
    # Load model
    model = ItemRanker.load(MODEL_PATH)

    # Load mappings
    champion_map, position_map, item_map = load_mappings()

    # Predict
    build = recommend_from_names(
        model,
        my_champion=my_champion,
        my_role=my_role,
        allies=allies,
        enemies=enemies,
        item_map=item_map,
        champion_map=champion_map,
        position_map=position_map,
    )
    print("Recommended build:", build)

def print_mappings_champion():
    champion_map, position_map, item_map = load_mappings()
    print("Champion Map:", champion_map)

def print_mappings_position():
    champion_map, position_map, item_map = load_mappings()
    print("Position Map:", position_map)

def print_mappings_item():
    champion_map, position_map, item_map = load_mappings()
    print("Item Map:", item_map)
    
def main():
    while True:
        print("=== MENU ===")
        print("0. Predict build with default values")
        print("1. Predict build")
        print("2. Train model")
        print("3. Print champions/positions/items mappings")
        print("4. Regenerate dataset")
        print("5. Exit")

        choice = input("Select: ").strip()
        if choice == "0":
            predict()
        if choice == "1":
            my_champion = input("Enter your champion: ").strip()
            my_role = input("Enter your role (TOP/JUNGLE/MID/ADC/UTILITY): ").strip()
            allies = input("Enter your allies (comma separated): ").strip().split(",")
            enemies = input("Enter your enemies (comma separated): ").strip().split(",")
            allies = [a.strip() for a in allies]
            enemies = [e.strip() for e in enemies]
            predict(my_champion=my_champion, my_role=my_role, allies=allies, enemies=enemies)
        elif choice == "2":
            choice = input("You are about to train and overwrite the model. It might take a while. Are you sure? (y/n)").strip()
            if choice == "y":
                train()
        elif choice == "3":
            print("1. Champion Map")
            print("2. Position Map")
            print("3. Item Map")
            sub_choice = input("Select: ").strip()
            if sub_choice == "1":
                print_mappings_champion()
            elif sub_choice == "2":
                print_mappings_position()
            elif sub_choice == "3":
                print_mappings_item()
            else:
                print("Invalid choice\n")
        elif choice == "4":
            choice = input("You are about to overwrite the dataset. Are you sure? (y/n)").strip()
            if choice == "y":
                run() # regenerate dataset and mappings
        elif choice == "5":
            print("Goodbye")
            break
        else:
            print("Invalid choice\n")


if __name__ == "__main__":
    main()
    
