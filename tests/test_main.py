import builtins
from unittest.mock import patch
import app.main as main


def test_menu_predict():
    # User selects: 1 → predict, then 5 → exit
    user_inputs = iter(["1", "5"])

    with patch.object(builtins, "input", lambda _: next(user_inputs)):
        with patch("app.main.predict") as mock_predict:
            with patch("app.main.train"):
                with patch("app.main.print_mappings_champion"):
                    with patch("app.main.print_mappings_position"):
                        with patch("app.main.print_mappings_item"):
                            with patch("app.main.run"):
                                main.main()
                                mock_predict.assert_called_once()


def test_menu_train_confirm_yes():
    # User selects: 2 → train, then confirms "y", then 5 → exit
    user_inputs = iter(["2", "y", "5"])

    with patch.object(builtins, "input", lambda _: next(user_inputs)):
        with patch("app.main.train") as mock_train:
            with patch("app.main.predict"):
                with patch("app.main.print_mappings_champion"):
                    with patch("app.main.print_mappings_position"):
                        with patch("app.main.print_mappings_item"):
                            with patch("app.main.run"):
                                main.main()
                                mock_train.assert_called_once()


def test_menu_train_confirm_no():
    # User selects: 2 → train, then "n" → cancel, then 5 → exit
    user_inputs = iter(["2", "n", "5"])

    with patch.object(builtins, "input", lambda _: next(user_inputs)):
        with patch("app.main.train") as mock_train:
            main.main()
            mock_train.assert_not_called()


def test_menu_print_champion_map():
    # User selects: 3 → mappings, then 1 → champion map, then 5 → exit
    user_inputs = iter(["3", "1", "5"])

    with patch.object(builtins, "input", lambda _: next(user_inputs)):
        with patch("app.main.print_mappings_champion") as mock_print:
            with patch("app.main.print_mappings_position"):
                with patch("app.main.print_mappings_item"):
                    main.main()
                    mock_print.assert_called_once()


def test_menu_print_position_map():
    user_inputs = iter(["3", "2", "5"])

    with patch.object(builtins, "input", lambda _: next(user_inputs)):
        with patch("app.main.print_mappings_position") as mock_print:
            with patch("app.main.print_mappings_champion"):
                with patch("app.main.print_mappings_item"):
                    main.main()
                    mock_print.assert_called_once()


def test_menu_print_item_map():
    user_inputs = iter(["3", "3", "5"])

    with patch.object(builtins, "input", lambda _: next(user_inputs)):
        with patch("app.main.print_mappings_item") as mock_print:
            with patch("app.main.print_mappings_champion"):
                with patch("app.main.print_mappings_position"):
                    main.main()
                    mock_print.assert_called_once()


def test_menu_regenerate_dataset_confirm_yes():
    user_inputs = iter(["4", "y", "5"])

    with patch.object(builtins, "input", lambda _: next(user_inputs)):
        with patch("app.main.run") as mock_run:
            main.main()
            mock_run.assert_called_once()


def test_menu_regenerate_dataset_confirm_no():
    user_inputs = iter(["4", "n", "5"])

    with patch.object(builtins, "input", lambda _: next(user_inputs)):
        with patch("app.main.run") as mock_run:
            main.main()
            mock_run.assert_not_called()


def test_menu_exit():
    user_inputs = iter(["5"])

    with patch.object(builtins, "input", lambda _: next(user_inputs)):
        with patch("app.main.predict") as p1, \
             patch("app.main.train") as p2, \
             patch("app.main.print_mappings_champion") as p3, \
             patch("app.main.print_mappings_position") as p4, \
             patch("app.main.print_mappings_item") as p5, \
             patch("app.main.run") as p6:
            main.main()
            p1.assert_not_called()
            p2.assert_not_called()
            p3.assert_not_called()
            p4.assert_not_called()
            p5.assert_not_called()
            p6.assert_not_called()
