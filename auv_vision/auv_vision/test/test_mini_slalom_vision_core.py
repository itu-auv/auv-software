import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "mini_slalom_core.py"
SPEC = importlib.util.spec_from_file_location("mini_slalom_vision_core", MODULE_PATH)
CORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORE)


def detection(class_id, x, height=300, confidence=0.9, y=360):
    return CORE.Detection(class_id, x, y, 30, height, confidence)


def test_left_direction_selects_white_left_of_red():
    target = CORE.select_target(
        [detection(3, 400), detection(2, 800)],
        1280,
        720,
        "left",
    )
    assert target.valid
    assert target.white_center_x < target.red_center_x
    assert abs(target.center_error + 0.0625) < 1e-6


def test_wrong_order_and_single_color_are_invalid():
    assert not CORE.select_target(
        [detection(3, 900), detection(2, 500)], 1280, 720, "left"
    ).valid
    assert not CORE.select_target([detection(2, 500)], 1280, 720, "left").valid


def test_largest_consistent_pair_is_selected():
    target = CORE.select_target(
        [
            detection(3, 100, height=80),
            detection(2, 300, height=80),
            detection(3, 500, height=320),
            detection(2, 900, height=300),
        ],
        1280,
        720,
        "left",
    )
    assert target.valid
    assert target.white_center_x == 500 / 1280
    assert target.red_center_x == 900 / 1280


def test_low_confidence_detection_is_rejected():
    target = CORE.select_target(
        [detection(3, 400, confidence=0.1), detection(2, 800)],
        1280,
        720,
        "left",
        min_confidence=0.25,
    )
    assert not target.valid


def test_vertically_inconsistent_cross_pair_is_not_selected():
    target = CORE.select_target(
        [
            detection(3, 200, height=330, y=100),
            detection(3, 500, height=300, y=360),
            detection(2, 850, height=320, y=360),
        ],
        1280,
        720,
        "left",
    )
    assert target.valid
    assert target.white_center_x == 500 / 1280
