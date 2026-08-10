# Operations for vitpose_process_node. One module per op type, each exporting
# create_op(params, ctx) -> op with process(frame) and optional
# draw(image_bgr, frame). The op contract is documented in
# vitpose_process_node.py's module docstring; auv_vision/VITPOSE_PLAN.md §5
# has the framework rationale. gate_pose.py (pose) and tetra_unfold.py
# (association + own image topic) are the two worked examples.
