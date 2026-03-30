import yaml
import numpy as np

with open(
    "/home/melih/catkin_ws/src/auv-software/auv_control/auv_control/config/taluy/default.yaml"
) as f:
    config = yaml.safe_load(f)

M = np.array(config["model"]["mass_inertia_matrix"])
D_lin = np.array(config["model"]["linear_damping_matrix"])
M_inv = np.linalg.inv(M)

print("M_inv diagonal:")
print(np.diag(M_inv))
print("M_inv[1,1]:", M_inv[1, 1])
