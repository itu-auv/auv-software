OBJECT_ID_MAP = {
    "gate": [0, 1],
    "pipe": [2, 3],
    "torpedo": [4],
    "torpedo_holes": [5],
    "bin": [6],
    "octagon": [7],
    "all": [0, 1, 2, 3, 4, 5, 6, 7],
    "none": [],
}

CAMERA_NAMES = {
    "front": "taluy/cameras/cam_front",
    "bottom": "taluy/cameras/cam_bottom",
}

CAMERA_FRAMES = {
    CAMERA_NAMES["front"]: "taluy/base_link/front_camera_optical_link_stabilized",
    CAMERA_NAMES["bottom"]: "taluy/base_link/bottom_camera_optical_link",
}

FRAME_ID_TO_CAMERA_NS = {
    "taluy/base_link/bottom_camera_link": CAMERA_NAMES["bottom"],
    "taluy/base_link/front_camera_link": CAMERA_NAMES["front"],
}

ID_TF_MAP = {
    CAMERA_NAMES["front"]: {
        0: "gate_sawfish_link",
        1: "gate_shark_link",
        2: "red_pipe_link",
        3: "white_pipe_link",
        4: "torpedo_map_link",
        5: "torpedo_hole_link",
        6: "bin_whole_link",
        7: "octagon_link",
    },
    CAMERA_NAMES["bottom"]: {
        0: "bin_shark_link",
        1: "bin_sawfish_link",
    },
}

# Detection IDs
SAWFISH_ID = 0
SHARK_ID = 1
RED_PIPE_ID = 2
WHITE_PIPE_ID = 3
TORPEDO_MAP_ID = 4
TORPEDO_HOLE_ID = 5
BIN_WHOLE_ID = 6
OCTAGON_ID = 7
BIN_SHARK_ID = 10
BIN_SAWFISH_ID = 11
