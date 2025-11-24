from dataclasses import dataclass
from typing import Optional

@dataclass
class PropConfig:
    id: int
    name: str
    real_height: Optional[float]
    real_width: Optional[float]

# Prop definitions
PROPS_CONFIG = {
    "sawfish": PropConfig(0, "sawfish", 0.3048, 0.3048),
    "shark": PropConfig(1, "shark", 0.3048, 0.3048),
    "red_pipe": PropConfig(2, "red_pipe", 0.90, None),
    "white_pipe": PropConfig(3, "white_pipe", 0.90, None),
    "torpedo_map": PropConfig(4, "torpedo_map", 0.6096, 0.6096),
    "torpedo_hole": PropConfig(5, "torpedo_hole", 0.125, 0.125),
    "bin_whole": PropConfig(6, "bin_whole", None, None),
    "octagon": PropConfig(7, "octagon", 0.74, 0.91),
    "bin_shark": PropConfig(10, "bin_shark", 0.30480, 0.30480),
    "bin_sawfish": PropConfig(11, "bin_sawfish", 0.30480, 0.30480),
}

# Mapping from link name to PropConfig
LINK_TO_PROP_MAP = {
    "gate_sawfish_link": PROPS_CONFIG["sawfish"],
    "gate_shark_link": PROPS_CONFIG["shark"],
    "red_pipe_link": PROPS_CONFIG["red_pipe"],
    "white_pipe_link": PROPS_CONFIG["white_pipe"],
    "torpedo_map_link": PROPS_CONFIG["torpedo_map"],
    "octagon_link": PROPS_CONFIG["octagon"],
    "bin_sawfish_link": PROPS_CONFIG["bin_sawfish"],
    "bin_shark_link": PROPS_CONFIG["bin_shark"],
    "torpedo_hole_shark_link": PROPS_CONFIG["torpedo_hole"],
    "torpedo_hole_sawfish_link": PROPS_CONFIG["torpedo_hole"],
}
