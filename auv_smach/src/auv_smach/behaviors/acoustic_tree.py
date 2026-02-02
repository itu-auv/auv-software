import py_trees
from .actions import AcousticTransmitBehavior, AcousticReceiveBehavior


def create_acoustic_tree(
    transmit_data: int = None,
    receive_expected: list = None,
    receive_timeout: float = None,
    enable_transmit: bool = True,
    enable_receive: bool = False,
):
    """
    Creates an Acoustic task behavior tree.

    This tree can be used for:
    - Transmitting acoustic data (status signals)
    - Receiving acoustic data (coordination with other AUVs)
    - Both transmit and receive in sequence

    Args:
        transmit_data: Data value to transmit (1-8). Required if enable_transmit=True.
        receive_expected: List of expected data values, or None to accept any.
        receive_timeout: Timeout in seconds for receiving (default: 30s).
        enable_transmit: Whether to include transmit behavior (default: True).
        enable_receive: Whether to include receive behavior (default: False).

    Returns:
        py_trees.composites.Sequence: The acoustic task tree.

    Example usage:
        # Just transmit
        tree = create_acoustic_tree(transmit_data=1)

        # Just receive
        tree = create_acoustic_tree(enable_transmit=False, enable_receive=True,
                                     receive_expected=[1, 2, 3], receive_timeout=30)

        # Transmit then receive
        tree = create_acoustic_tree(transmit_data=1, enable_receive=True,
                                     receive_expected=[1, 2], receive_timeout=60)
    """

    root = py_trees.composites.Sequence("AcousticTask", memory=True)

    # Add transmit behavior if enabled
    if enable_transmit:
        if transmit_data is None:
            raise ValueError(
                "transmit_data must be specified when enable_transmit=True"
            )
        root.add_child(
            AcousticTransmitBehavior(
                name="TransmitAcoustic",
                acoustic_data=transmit_data,
            )
        )

    # Add receive behavior if enabled
    if enable_receive:
        root.add_child(
            AcousticReceiveBehavior(
                name="ReceiveAcoustic",
                expected_data=receive_expected,
                timeout=receive_timeout,
            )
        )

    return root
