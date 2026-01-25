import py_trees
from .actions import (
    SetBoolServiceBehavior,
    DelayBehavior,
    AlignFrameBehavior,
)


def create_coin_flip_subtree(name: str):
    """
    Creates the 'Coin Flip' maneuver subtree.
    SMACH equivalent: CoinFlipState (coin_flip.py)
    """
    root = py_trees.composites.Sequence(name=name, memory=True)

    # 1. Enable Rescue Coin Flip Service
    root.add_child(
        SetBoolServiceBehavior(
            name="EnableRescuer", service_name="toggle_coin_flip_rescuer", value=True
        )
    )

    # 2. Wait for frame
    root.add_child(DelayBehavior("WaitForRescueFrame", duration=1.0))

    # 3. Disable Rescue Coin Flip Service
    root.add_child(
        SetBoolServiceBehavior(
            name="DisableRescuer", service_name="toggle_coin_flip_rescuer", value=False
        )
    )

    # 4. Align to frame (Keep Orientation = True)
    root.add_child(
        AlignFrameBehavior(
            name="AlignToRescuerPos",
            source_frame="taluy/base_link",
            target_frame="coin_flip_rescuer",
            dist_threshold=0.1,
            yaw_threshold=0.1,
            confirm_duration=1.0,
            timeout=15.0,
            cancel_on_success=False,
            keep_orientation=True,
            max_linear_velocity=0.3,
        )
    )

    # 5. Align to frame (Keep Orientation = False)
    root.add_child(
        AlignFrameBehavior(
            name="AlignToRescuerFull",
            source_frame="taluy/base_link",
            target_frame="coin_flip_rescuer",
            dist_threshold=0.1,
            yaw_threshold=0.1,
            confirm_duration=1.0,
            timeout=15.0,
            cancel_on_success=False,
            keep_orientation=False,
        )
    )

    return root
