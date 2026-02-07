import py_trees
from .actions import (
    SetDepthBehavior,
    SetDetectionFocusBehavior,
    CancelAlignControllerBehavior,
    SetBoolServiceBehavior,
    AlignFrameBehavior,
)

from .subtrees import (
    create_search_subtree,
    create_look_around_subtree,
    create_dynamic_path_subtree,
)
from .coin_flip import create_coin_flip_subtree
from .roll import create_roll_subtree, create_yaw_subtree


def create_gate_tree(
    gate_depth: float,
    gate_search_depth: float,
    roll_depth: float,
    gate_exit_angle: float,
    enable_roll: bool,
    enable_yaw: bool,
    enable_coin_flip: bool,
):
    """
    Creates the Gate task behavior tree.

    Args:
        gate_depth: Target depth for passing through the gate.
        gate_search_depth: Target depth for searching/looking at gate.
        roll_depth: Target depth for performing the roll maneuver.
        gate_exit_angle: Angle offset for alignment after exiting.
        enable_roll: Whether to perform the California Roll maneuver.
        enable_yaw: Whether to perform the Two Yaw maneuver (if roll is disabled).
        enable_coin_flip: Whether to perform the Coin Flip maneuver.
    """

    root = py_trees.composites.Sequence("GateTask", memory=True)

    # 1. Initial depth
    root.add_child(SetDepthBehavior("SetInitialDepth", depth=-0.5, sleep_duration=3.0))

    # 2. Enable gate trajectory publisher
    root.add_child(
        SetBoolServiceBehavior(
            "EnableTrajectory", service_name="toggle_gate_trajectory", value=True
        )
    )

    # 3. Optional: Coin Flip
    if enable_coin_flip:
        root.add_child(create_coin_flip_subtree("CoinFlipManeuver"))

    # 4. Set detection focus to gate
    root.add_child(SetDetectionFocusBehavior("FocusGate", focus_object="gate"))

    # 5. Roll depth
    root.add_child(
        SetDepthBehavior("SetRollDepth", depth=roll_depth, sleep_duration=3.0)
    )

    # 6. Find and aim gate (SearchForProp)
    root.add_child(
        create_search_subtree(
            name="FindAndAimGate",
            source_frame="taluy/base_link",
            look_at_frame="gate_middle_part",
            alignment_frame="gate_search",
            rotation_speed=0.2,
        )
    )

    # 7. Optional: Roll OR Yaw (Priority: Roll > Yaw)
    if enable_roll:
        # California Roll
        root.add_child(
            create_roll_subtree(
                name="CaliforniaRollManeuver",
                gate_look_at_frame="gate_middle_part",
                roll_torque=50.0,
            )
        )
    elif enable_yaw:
        # Two Yaw
        root.add_child(
            create_yaw_subtree(
                name="TwoYawManeuver",
                yaw_frame="gate_search",  # Checks gate_tree.py vs gate.py parity - gate_search matches
            )
        )

    # 8. Set gate trajectory depth (gate_search_depth)
    root.add_child(
        SetDepthBehavior(
            "SetGateTrajectoryDepth", depth=gate_search_depth, sleep_duration=3.0
        )
    )

    # 9. Look at gate (SearchForProp)
    root.add_child(
        create_search_subtree(
            name="LookAtGate",
            source_frame="taluy/base_link",
            look_at_frame="gate_middle_part",
            alignment_frame="gate_search",
            rotation_speed=0.2,
            duration=3.0,
        )
    )

    # 10. Check Surroundings (Selam)
    root.add_child(
        create_look_around_subtree(
            name="LookAround",
            target_frame_name="selam_frame",
            source_frame="taluy/base_link",
            reference_frame="odom",
            look_angle=0.5,
            timeout=10.0,
        )
    )

    # 11. Look at gate for trajectory (SearchForProp)
    root.add_child(
        create_search_subtree(
            name="LookAtGateForTrajectory",
            source_frame="taluy/base_link",
            look_at_frame="gate_middle_part",
            alignment_frame="gate_search",
            rotation_speed=0.2,
            duration=7.0,
        )
    )

    # 12. Disable trajectory
    root.add_child(
        SetBoolServiceBehavior(
            "DisableTrajectory", service_name="toggle_gate_trajectory", value=False
        )
    )

    # 13. Focus None
    root.add_child(SetDetectionFocusBehavior("FocusNone", focus_object="none"))

    # 14. Set Gate Depth
    root.add_child(
        SetDepthBehavior("SetGateDepth", depth=gate_depth, sleep_duration=3.0)
    )

    # 15. Dynamic Path to Entrance
    root.add_child(
        create_dynamic_path_subtree(
            name="PathToEntrance", plan_target_frame="gate_entrance"
        )
    )

    # 16. Dynamic Path to Exit
    root.add_child(
        create_dynamic_path_subtree(name="PathToExit", plan_target_frame="gate_exit")
    )

    # 17. Align after exit
    root.add_child(
        AlignFrameBehavior(
            name="AlignAfterExit",
            source_frame="taluy/base_link",
            target_frame="gate_exit",
            angle_offset=gate_exit_angle,
            yaw_threshold=0.25,  # Relaxed from default 0.1 to allow convergence
            timeout=20.0,  # Increased from 10.0 for more alignment time
            confirm_duration=1.0,
            cancel_on_success=True,
            keep_orientation=False,  # Explicitly False matching SMACH default in AlignFrameState
        )
    )

    # Final Cleanup
    root.add_child(CancelAlignControllerBehavior("CancelAlign"))

    return root
