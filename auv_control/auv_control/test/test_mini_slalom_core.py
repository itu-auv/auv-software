import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "mini_slalom_core.py"
SPEC = importlib.util.spec_from_file_location("mini_slalom_core", MODULE_PATH)
CORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CORE)


def test_alignment_controls_forward_force_and_yaw_direction():
    config = CORE.ControllerConfig(yaw_kp=10.0, yaw_kd=0.0, yaw_sign=-1.0)
    controller = CORE.SlalomController(config)
    controller.reset("left", 3)

    aligned = controller.update(1.0, CORE.Target(valid=True, center_error=0.1))
    assert aligned.state == "ADVANCE"
    assert aligned.force_x == config.forward_force
    assert aligned.torque_z == -1.0

    misaligned = controller.update(2.0, CORE.Target(valid=True, center_error=0.5))
    assert misaligned.state == "ALIGN"
    assert misaligned.force_x == 0.0
    assert abs(misaligned.torque_z) <= config.max_yaw_torque


def test_near_gate_loss_counts_pass_and_finishes_after_configured_count():
    config = CORE.ControllerConfig(
        near_height_ratio=0.5,
        near_confirm_duration=0.5,
        pass_confirm_duration=0.8,
        pass_forward_force=7.0,
        exit_duration=0.5,
    )
    controller = CORE.SlalomController(config)
    controller.reset("left", 1)
    near = CORE.Target(valid=True, center_error=0.05, gate_height_ratio=0.7)

    controller.update(0.0, near)
    armed = controller.update(0.6, near)
    assert armed.state == "PASS_ARMED"

    confirming = controller.update(0.7, CORE.Target())
    assert confirming.state == "PASS_CONFIRM"
    assert confirming.force_x == 7.0
    assert confirming.torque_z == 0.0

    command = controller.update(1.6, CORE.Target())
    assert command.state == "EXIT"
    assert command.gates_passed == 1
    assert not command.finished

    command = controller.update(2.2, CORE.Target())
    assert command.finished
    assert command.state == "FINISHED"


def test_single_near_frame_does_not_arm_or_count_pass():
    controller = CORE.SlalomController(
        CORE.ControllerConfig(
            near_height_ratio=0.5,
            near_confirm_duration=0.5,
            pass_confirm_duration=0.5,
        )
    )
    controller.reset("left", 3)
    controller.update(
        0.0, CORE.Target(valid=True, center_error=0.0, gate_height_ratio=0.7)
    )
    command = controller.update(1.0, CORE.Target())
    assert command.gates_passed == 0
    assert command.state == "SEARCH"


def test_unaligned_near_gate_does_not_arm():
    controller = CORE.SlalomController(
        CORE.ControllerConfig(
            align_error_threshold=0.2,
            near_height_ratio=0.5,
            near_confirm_duration=0.4,
        )
    )
    controller.reset("left", 3)
    near_but_unaligned = CORE.Target(
        valid=True, center_error=0.4, gate_height_ratio=0.8
    )
    controller.update(0.0, near_but_unaligned)
    command = controller.update(0.8, near_but_unaligned)
    assert command.state == "ALIGN"
    assert not controller.pass_armed


def test_detection_return_cancels_pass_confirmation():
    controller = CORE.SlalomController(
        CORE.ControllerConfig(
            near_height_ratio=0.5,
            near_confirm_duration=0.4,
            pass_confirm_duration=0.8,
        )
    )
    controller.reset("left", 3)
    near = CORE.Target(valid=True, center_error=0.0, gate_height_ratio=0.8)

    controller.update(0.0, near)
    controller.update(0.5, near)
    assert controller.update(0.6, CORE.Target()).state == "PASS_CONFIRM"

    returned = controller.update(1.0, near)
    assert returned.gates_passed == 0
    assert returned.state == "ADVANCE"
    assert not controller.pass_armed

    # A later disappearance cannot count until the aligned-near dwell has
    # armed the gate again.
    controller.update(1.1, CORE.Target())
    command = controller.update(2.0, CORE.Target())
    assert command.gates_passed == 0
    assert command.state == "SEARCH"


def test_wrench_overlay_preserves_depth_roll_pitch():
    nominal = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    visual = (10.0, 0.0, 0.0, 0.0, 0.0, -7.0)
    assert CORE.overlay_wrench(nominal, visual, False, True) == nominal
    assert CORE.overlay_wrench(nominal, visual, True, True) == (
        10.0,
        0.0,
        3.0,
        4.0,
        5.0,
        -7.0,
    )
    assert CORE.overlay_wrench(nominal, visual, True, False) == (
        0.0,
        0.0,
        3.0,
        4.0,
        5.0,
        0.0,
    )
