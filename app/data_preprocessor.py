import pandas as pd
from .config import ITEM_COLS, EXCLUDED_ITEMS

def build_mappings(df):
    CHAMPION_TO_INDEX = {champion: idx for idx, champion in enumerate(df['champion'].unique())}
    POSITION_TO_INDEX = {
        "TOP": 0, "JUNGLE": 1, "MID": 2, "ADC": 3, "UTILITY": 4
        }

    all_items = pd.unique(df[ITEM_COLS].values.ravel())
    all_items = [item for item in all_items if item != 0 and item not in EXCLUDED_ITEMS]

    ITEM_TO_INDEX = {item: idx for idx, item in enumerate(all_items)}

    return CHAMPION_TO_INDEX, POSITION_TO_INDEX, ITEM_TO_INDEX


class DataPreprocessor:
    def __init__(self, df, champion_map, position_map, item_map):
        self.df = df
        self.CHAMPION_TO_INDEX = champion_map
        self.POSITION_TO_INDEX = position_map
        self.ITEM_TO_INDEX = item_map

    def preprocess(self):
        df = self.df.copy()
        
        df['champion_idx'] = df['champion'].map(self.CHAMPION_TO_INDEX).fillna(-1).astype(int)
        df['position_idx'] = df['position'].map(self.POSITION_TO_INDEX).fillna(-1).astype(int)

        for item_col in ITEM_COLS:
            # if item is not in ITEM_TO_INDEX, assign -1
            df[item_col + "_idx"] = df[item_col].map(lambda x: self.ITEM_TO_INDEX.get(x, -1)).fillna(-1).astype(int)
        
        return df
    