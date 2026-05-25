from app.mapping_loader import load_mappings

def test_mappings_load():
    champion_map, item_map, position_map = load_mappings()

    assert isinstance(champion_map, dict)
    assert isinstance(item_map, dict)
    assert isinstance(position_map, dict)

    assert len(champion_map) > 0
    assert len(item_map) > 0
    assert len(position_map) > 0
