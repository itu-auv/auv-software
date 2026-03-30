import numpy as np

M = np.array(
    [
        [29.8419, 0, 0, 0, -1.5306, -0.4401],
        [0, 29.8419, 0, 1.5306, 0, -0.6235],
        [0, 0, 29.8419, 0.4401, 0.6235, 0],
        [0, 1.5306, 0.4401, 0.9430, -0.0317, 0.1290],
        [-1.5306, 0, 0.6235, -0.0317, 2.1200, -0.1070],
        [-0.4401, -0.6235, 0, 0.1290, -0.1070, 2.1800],
    ]
)
M_inv = np.linalg.inv(M)

D_lin = np.array(
    [
        [62.8822, 0, 0, 0, 0, 0],
        [0, 106.733, 0, 0, 0, 0],
        [0, 0, 89.2, 0, 0, 0],
        [0, 0, 0, 0.0, 0, 0],
        [0, 0, 0, 0, 0.0, 0],
        [0, 0, 0, 0, 0, 4.04997],
    ]
)

D_quad = np.array(
    [
        [116.822, 0, 0, 0, 0, 0],
        [0, 82.556, 0, 0, 0, 0],
        [0, 0, 136.16, 0, 0, 0],
        [0, 0, 0, 0.0, 0, 0],
        [0, 0, 0, 0, 0.0, 0],
        [0, 0, 0, 0, 0, 2.41789],
    ]
)


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def calculate_coriolis_matrix(v):
    C = np.zeros((6, 6))
    M11 = M[0:3, 0:3]
    M12 = M[0:3, 3:6]
    M21 = M[3:6, 0:3]
    M22 = M[3:6, 3:6]
    v1 = v[0:3]
    v2 = v[3:6]
    Mv1 = np.dot(M11, v1) + np.dot(M12, v2)
    Mv2 = np.dot(M21, v1) + np.dot(M22, v2)
    s_mv1 = skew(Mv1)
    s_mv2 = skew(Mv2)
    C[0:3, 3:6] = -s_mv1
    C[3:6, 0:3] = -s_mv1
    C[3:6, 3:6] = -s_mv2
    return C


v = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # Drift left
dt = 0.1

for i in range(10):
    # Simulated controller (only P term for simplicity, Kp = 1.0)
    desired_v = 0.0
    err = desired_v - v[1]
    pid_force = 29.8419 * (1.0 * err)  # kp_vy=1.0 * mass
    wrench = np.array([0, pid_force, 0, 0, 0, 0])

    # Model
    d_lin = np.dot(D_lin, v)
    d_quad = np.dot(D_quad, np.abs(v) * v)
    C_mat = calculate_coriolis_matrix(v)
    c_f = np.dot(C_mat, v)

    net_force = wrench - d_lin - d_quad - c_f
    accel = np.dot(M_inv, net_force)
    v = v + accel * dt
    print(f"{i}: v_y = {v[1]:.4f}, wrench_y = {wrench[1]:.2f}")
