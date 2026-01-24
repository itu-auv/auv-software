import py_trees
from .actions import (
    RotateBehavior,
    SetFrameLookingAtBehavior,
    CreateFrameAtCurrentPositionBehavior,
    AlignFrameBehavior,
    PlanPathBehavior,
    ExecutePathBehavior,
    TriggerServiceBehavior,
    CancelAlignControllerBehavior,
)
from .conditions import IsTransformAvailable


def create_search_subtree(
    name: str,
    source_frame: str,
    look_at_frame: str,
    alignment_frame: str,
    rotation_speed: float = 0.2,
    timeout: float = 25.0,
    duration: float = 3.0,
):
    """
    Creates a subtree for searching and aligning to a prop (e.g., gate).

    Structure:
        Selector
        ├── IsTransformAvailable  (TF already visible?)
        └── Sequence
            ├── RotateBehavior          (Rotate until TF found)
            └── SetFrameLookingAtBehavior (Set facing direction)

    SMACH equivalent: SearchForPropState (common.py)
    """

    # Root: Selector - "Either TF is already available, OR search for it"
    root = py_trees.composites.Selector(name, memory=False)

    # Child 1: Check if TF already available (skip rotation if yes)
    root.add_child(
        IsTransformAvailable(
            name=f"{name}_CheckTF",
            source_frame=source_frame,
            target_frame=look_at_frame,
        )
    )

    # Child 2: Sequence - Rotate then set facing
    search_sequence = py_trees.composites.Sequence(f"{name}_Search", memory=True)

    search_sequence.add_child(
        RotateBehavior(
            name=f"{name}_Rotate",
            source_frame=source_frame,
            look_at_frame=look_at_frame,
            rotation_speed=rotation_speed,
            timeout=timeout,
        )
    )

    search_sequence.add_child(
        SetFrameLookingAtBehavior(
            name=f"{name}_LookAt",
            source_frame=source_frame,
            look_at_frame=look_at_frame,
            alignment_frame=alignment_frame,
            duration=duration,
        )
    )

    root.add_child(search_sequence)

    return root


def create_look_around_subtree(
    name: str,
    target_frame_name: str = "selam_frame",
    source_frame: str = "taluy/base_link",
    reference_frame: str = "odom",
    look_angle: float = 0.5,
    timeout: float = 10.0,
):
    """
    Creates a subtree for 'Looking Around' (Selam Movement).

    Structure:
        Sequence
        ├── CreateFrameAtCurrentPosition (Set reference frame)
        ├── AlignFrame (Look Left)
        ├── AlignFrame (Look Right)
        └── AlignFrame (Look Center)

    SMACH equivalent: LookAroundState (common.py:1177-1250)
    """
    root = py_trees.composites.Sequence(name, memory=True)

    # 1. Create Reference Frame
    root.add_child(
        CreateFrameAtCurrentPositionBehavior(
            name=f"{name}_CreateFrame",
            target_frame_name=target_frame_name,
            source_frame=source_frame,
            reference_frame=reference_frame,
        )
    )

    # 2. Look Left (+angle)
    # Note: Disable heading control during movement, re-enable at the very end
    root.add_child(
        AlignFrameBehavior(
            name=f"{name}_LookLeft",
            source_frame=source_frame,
            target_frame=target_frame_name,
            angle_offset=look_angle,
            timeout=timeout,
            heading_control=False,
            enable_heading_control_afterwards=False,
        )
    )

    # 3. Look Right (-angle)
    root.add_child(
        AlignFrameBehavior(
            name=f"{name}_LookRight",
            source_frame=source_frame,
            target_frame=target_frame_name,
            angle_offset=-look_angle,
            timeout=timeout,
            heading_control=False,
            enable_heading_control_afterwards=False,
        )
    )

    # 4. Look Center (0.0) & Restore Heading Control
    root.add_child(
        AlignFrameBehavior(
            name=f"{name}_LookCenter",
            source_frame=source_frame,
            target_frame=target_frame_name,
            angle_offset=0.0,
            timeout=timeout,
            heading_control=False,
            enable_heading_control_afterwards=True,  # Important: Re-enable HC here!
        )
    )

    return root


def create_dynamic_path_subtree(
    name: str,
    plan_target_frame: str,
    align_source_frame: str = "taluy/base_link",
    align_target_frame: str = "dynamic_target",
    angle_offset: float = 0.0,
):
    """
    Creates a subtree for Dynamic Path Planning & Execution.

    Structure:
        Sequence
        ├── PlanPath (Call /set_plan)
        ├── AlignFrame (Wait=False, Cancel=False) -> Start aligning to dynamic_target
        ├── ExecutePath (Execute path action)
        └── TriggerService (Stop planning)

    SMACH equivalent: DynamicPathState (common.py:1120-1175)
    """
    # Root is a Selector: Try execution, if it fails, cleanup!
    # This acts like a try-finally block for safety.
    root = py_trees.composites.Selector(name, memory=True)

    # 1. Main Execution Sequence
    main_sequence = py_trees.composites.Sequence(f"{name}_Main", memory=True)
    root.add_child(main_sequence)

    # 1.1 Plan Path
    main_sequence.add_child(
        PlanPathBehavior(
            name=f"{name}_Plan",
            target_frame=plan_target_frame,
            angle_offset=angle_offset,
        )
    )

    # 1.2 Start Alignment (Fire and Forget)
    # We want alignment to run IN PARALLEL with execution, so we don't wait.
    # We also don't cancel it on success of THIS node (it just keeps aligning).
    main_sequence.add_child(
        AlignFrameBehavior(
            name=f"{name}_StartAlign",
            source_frame=align_source_frame,
            target_frame=align_target_frame,
            wait_for_alignment=False,  # Don't block
            cancel_on_success=False,  # Don't stop controller
            heading_control=True,  # Keep existing logic
        )
    )

    # 1.3 Execute Path
    main_sequence.add_child(ExecutePathBehavior(name=f"{name}_Execute"))

    # 1.4 Stop Planning (Normal finish)
    main_sequence.add_child(
        TriggerServiceBehavior(
            name=f"{name}_StopPlanning", service_name="/stop_planning"
        )
    )

    # 2. Cleanup Sequence (Runs if Main fails)
    # If main sequence fails (e.g. execution error), Selector goes here.
    cleanup_sequence = py_trees.composites.Sequence(f"{name}_Cleanup", memory=True)
    root.add_child(cleanup_sequence)

    # 2.1 Stop Planning
    cleanup_sequence.add_child(
        TriggerServiceBehavior(
            name=f"{name}_EmergencyStopPlan", service_name="/stop_planning"
        )
    )

    # 2.2 Stop Alignment (Crucial! Don't leave robot spinning)
    cleanup_sequence.add_child(
        CancelAlignControllerBehavior(name=f"{name}_EmergencyStopAlign")
    )

    # 2.3 Explicitly return FAILURE to indicate the task failed (even if cleanup succeeded)
    # This matches SMACH 'aborted' outcome.
    cleanup_sequence.add_child(py_trees.behaviours.Failure(name=f"{name}_TaskFailed"))

    return root
