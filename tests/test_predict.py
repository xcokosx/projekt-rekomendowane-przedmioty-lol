import numpy as np
from unittest.mock import MagicMock, patch
import app.predict as pred


def test_build_team_vectors_from_names():
    champion_map = {"Jinx": 0, "Lux": 1, "Ahri": 2}

    allies = ["Jinx", "Ahri"]
    enemies = ["Lux"]

    ally_vec, enemy_vec = pred.build_team_vectors_from_names(allies, enemies, champion_map)

    assert ally_vec.tolist() == [1.0, 0.0, 1.0]
    assert enemy_vec.tolist() == [0.0, 1.0, 0.0]


def test_recommend_from_names_basic():
    # Fake mappings
    champion_map = {"Jinx": 0}
    position_map = {"BOTTOM": 0}
    item_map = {0: "ItemA", 1: "ItemB", 2: "ItemC"}

    # Fake model that always returns increasing scores
    fake_model = MagicMock()
    fake_model.forward.side_effect = [
        np.array([0.1, 0.9, 0.2]),  # pick item 1
        np.array([0.1, -1.0, 0.8]), # item 1 masked, pick item 2
        np.array([0.9, -1.0, -1.0]),# pick item 0
        np.array([0.9, -1.0, -1.0]),
        np.array([0.9, -1.0, -1.0]),
        np.array([0.9, -1.0, -1.0]),
    ]

    with patch("app.predict.one_hot", return_value=np.array([1.0])):
        build = pred.recommend_from_names(
            fake_model,
            my_champion="Jinx",
            my_role="BOTTOM",
            allies=[],
            enemies=[],
            item_map=item_map,
            champion_map=champion_map,
            position_map=position_map,
        )

    # First 3 picks are deterministic from side_effect
    assert build[0] == 1
    assert build[1] == 2
    assert build[2] == 0

    # Remaining picks must be valid indices
    assert all(isinstance(i, int) for i in build)
    assert len(build) == 6


def test_recommend_masks_already_picked_items():
    champion_map = {"Jinx": 0}
    position_map = {"BOTTOM": 0}
    item_map = {0: "A", 1: "B", 2: "C"}

    # Model always returns same scores
    fake_model = MagicMock()
    fake_model.forward.return_value = np.array([0.9, 0.8, 0.7])

    with patch("app.predict.one_hot", return_value=np.array([1.0])):
        build = pred.recommend_from_names(
            fake_model,
            my_champion="Jinx",
            my_role="BOTTOM",
            allies=[],
            enemies=[],
            item_map=item_map,
            champion_map=champion_map,
            position_map=position_map,
        )

    # Should pick items in descending score order: 0 → 1 → 2 → then repeat masked pattern
    assert build[0] == 0
    assert build[1] == 1
    assert build[2] == 2

    # After all items are used, the model will pick the highest non-masked again
    assert build[3] in [0, 1, 2]
    assert build[4] in [0, 1, 2]
    assert build[5] in [0, 1, 2]


def test_recommend_raises_on_unknown_champion():
    champion_map = {"Jinx": 0}
    position_map = {"BOTTOM": 0}
    item_map = {0: "A"}

    fake_model = MagicMock()

    with patch("app.predict.one_hot", return_value=np.array([1.0])):
        try:
            pred.recommend_from_names(
                fake_model,
                my_champion="UnknownChamp",
                my_role="BOTTOM",
                allies=[],
                enemies=[],
                item_map=item_map,
                champion_map=champion_map,
                position_map=position_map,
            )
            assert False, "Expected KeyError"
        except KeyError:
            pass


def test_recommend_raises_on_unknown_role():
    champion_map = {"Jinx": 0}
    position_map = {"BOTTOM": 0}
    item_map = {0: "A"}

    fake_model = MagicMock()

    with patch("app.predict.one_hot", return_value=np.array([1.0])):
        try:
            pred.recommend_from_names(
                fake_model,
                my_champion="Jinx",
                my_role="ADC",  # invalid
                allies=[],
                enemies=[],
                item_map=item_map,
                champion_map=champion_map,
                position_map=position_map,
            )
            assert False, "Expected KeyError"
        except KeyError:
            pass
