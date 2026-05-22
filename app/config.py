FILE_PATH = "../dataset/matches.parquet" # path to the dataset file, must be parquet format

ITEM_COLS = ["item0", "item1", "item2", "item3", "item4", "item5", "item6"] # item6 is trinket
EXCLUDED_ITEMS = {
    2055, 2056, 3340, 4638, 4643, #trinkets
    2138, 2139, 2140, 2150, 2151, 2152, # elixirs

    }