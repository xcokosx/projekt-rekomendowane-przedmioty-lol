from app.config import FILE_PATH,MODEL_PATH,ITEM_COLS,EXCLUDED_ITEMS

def test_config():
    assert isinstance(FILE_PATH, str) and FILE_PATH.endswith(".parquet")
    assert isinstance(MODEL_PATH, str) and MODEL_PATH.endswith(".npz")
    assert isinstance(ITEM_COLS, list) and all(isinstance(col, str) for col in ITEM_COLS)
    assert isinstance(EXCLUDED_ITEMS, set) and all(isinstance(item, int) for item in EXCLUDED_ITEMS)