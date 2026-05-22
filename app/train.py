import numpy as np
import pandas as pd
from .config import FILE_PATH
from .data_preprocessor import DataPreprocessor, build_mappings
from .dataset_builder import build_recommender_dataset
from .neural_network import ItemRanker

def train_model(epochs=10, batch_size=32, lr=0.001, lr_decay=0.999, val_split=0.1, seed=None):
    if seed is None:
        seed = 42 # default seed for reproducibility
    df_raw = pd.read_parquet(FILE_PATH)
    champion_map, position_map, item_map = build_mappings(df_raw)
    preprocessor = DataPreprocessor(df_raw, champion_map, position_map, item_map)
    df = preprocessor.preprocess()
    
    #build set
    X, Y = build_recommender_dataset(df, champion_map, item_map, position_map)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))
    X, Y = X[indices], Y[indices]

    split = int(len(X) * (1 - val_split))
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    model = ItemRanker(n=X.shape[1], num_items=len(item_map), hidden=64, seed=seed)

    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        #shuffle training data each epoch
        indices = rng.permutation(len(X_train))
        X_train, Y_train = X_train[indices], Y_train[indices]

        for i in range(0, len(X_train), batch_size):
            xb = X_train[i:i+batch_size]
            yb = Y_train[i:i+batch_size]

            for x_i, y_i in zip(xb, yb):
                pred = model.forward(x_i)
                model.backward(pred, y_i, lr=lr)
        
        lr *= lr_decay

        val_loss = 0.0
        for x_i, y_i in zip(X_val, Y_val):
            pred = model.forward(x_i)
            val_loss += ItemRanker.bce_loss(pred, y_i)
        val_loss /= len(X_val)

        print(f"Epoch {epoch+1}/{epochs} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            model.save("dataset/best_model.npz")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

    np.savez(
    "dataset/mappings.npz",
    champion_map=champion_map,
    item_map=item_map,
    position_map=position_map
    )

    print("Training complete")
    return model
