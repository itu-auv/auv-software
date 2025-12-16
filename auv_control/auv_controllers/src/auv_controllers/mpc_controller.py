from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import casadi as ca


# ----------------------------
# Reuse your VehicleParams as-is
# ----------------------------
@dataclass
class VehicleParams:
    m: float
    rg: np.ndarray
    I_g: np.ndarray

    MA: np.ndarray
    DL: np.ndarray
    DQ: np.ndarray

    W: float
    B: float
    rb: np.ndarray


# ----------------------------
# Numeric helpers (NumPy) for constant matrices
# ----------------------------
def skew_np(a: np.ndarray) -> np.ndarray:
    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
    return np.array([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]], dtype=float)


def mass_rigid_body_np(p: VehicleParams) -> np.ndarray:
    m, rg, I_g = (
        float(p.m),
        np.asarray(p.rg, float).reshape(3),
        np.asarray(p.I_g, float).reshape(3, 3),
    )
    Srg = skew_np(rg)

    # Inertia about body origin O (parallel axis)
    I_o = I_g - m * (Srg @ Srg)

    MRB = np.zeros((6, 6), dtype=float)
    MRB[0:3, 0:3] = m * np.eye(3)
    MRB[0:3, 3:6] = -m * Srg
    MRB[3:6, 0:3] = m * Srg
    MRB[3:6, 3:6] = I_o
    return MRB


def quat_normalize_np(q: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.asarray(q, float).reshape(4)
    n = np.linalg.norm(q)
    if n < eps:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return q / n


# ----------------------------
# CasADi symbolic helpers
# ----------------------------
def skew_ca(a: ca.MX) -> ca.MX:
    ax, ay, az = a[0], a[1], a[2]
    return ca.vertcat(
        ca.horzcat(0, -az, ay),
        ca.horzcat(az, 0, -ax),
        ca.horzcat(-ay, ax, 0),
    )


def quat_mult_ca(q1: ca.MX, q2: ca.MX) -> ca.MX:
    # Hamilton, [x,y,z,w]
    x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
    x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]
    return ca.vertcat(
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def quat_conj_ca(q: ca.MX) -> ca.MX:
    # For unit quaternion: inverse == conjugate
    return ca.vertcat(-q[0], -q[1], -q[2], q[3])


def quat_to_rotmat_ca(q: ca.MX) -> ca.MX:
    x, y, z, w = q[0], q[1], q[2], q[3]
    return ca.vertcat(
        ca.horzcat(1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        ca.horzcat(2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        ca.horzcat(2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def omega_to_quat_rate_ca(q: ca.MX, omega_b: ca.MX) -> ca.MX:
    omega_quat = ca.vertcat(omega_b[0], omega_b[1], omega_b[2], 0.0)
    return 0.5 * quat_mult_ca(q, omega_quat)


def coriolis_from_mass_ca(M: ca.DM, nu: ca.MX) -> ca.MX:
    # Fossen-style, M constant 6x6
    p = M @ nu
    p_lin = p[0:3]
    p_ang = p[3:6]
    Z3 = ca.MX.zeros(3, 3)
    return ca.vertcat(
        ca.horzcat(Z3, -skew_ca(p_lin)),
        ca.horzcat(-skew_ca(p_lin), -skew_ca(p_ang)),
    )


# ----------------------------
# CasADi AUV model (continuous + discrete RK4)
# ----------------------------
class UnderwaterVehicle6DOF_CasADi:
    """
    State x = [pos(3); quat(4); nu(6)]   (13x1)
    Control u = tau (6x1) body wrench
    """

    nx = 13
    nu = 6

    def __init__(self, p: VehicleParams, dt: float, quat_norm_eps: float = 1e-9):
        self.p = p
        self.dt = float(dt)
        self.quat_norm_eps = float(quat_norm_eps)

        # Precompute constant mass matrices numerically
        MRB_np = mass_rigid_body_np(p)
        M_np = MRB_np + np.asarray(p.MA, float).reshape(6, 6)

        # Constant inverse
        Minv_np = np.linalg.inv(M_np)

        # Store constants as CasADi DM
        self.MRB = ca.DM(MRB_np)
        self.MA = ca.DM(np.asarray(p.MA, float).reshape(6, 6))
        self.Minv = ca.DM(Minv_np)

        self.DL = ca.DM(np.asarray(p.DL, float).reshape(6, 6))
        self.DQ = ca.DM(np.asarray(p.DQ, float).reshape(6, 6))

        self.rg = ca.DM(np.asarray(p.rg, float).reshape(3, 1))
        self.rb = ca.DM(np.asarray(p.rb, float).reshape(3, 1))
        self.W = float(p.W)
        self.B = float(p.B)

        self._build_functions()

    def _restoring(self, quat: ca.MX) -> ca.MX:
        # g(q) in BODY frame for ENU coords (same as your numpy version)
        R = quat_to_rotmat_ca(quat)

        fg_n = ca.DM([0.0, 0.0, -self.W])  # ENU
        fb_n = ca.DM([0.0, 0.0, self.B])  # ENU

        fg_b = R.T @ fg_n
        fb_b = R.T @ fb_n

        g_force = -(fg_b + fb_b)
        g_moment = -(ca.cross(self.rg, fg_b) + ca.cross(self.rb, fb_b))

        return ca.vertcat(g_force, g_moment)

    def _damping(self, nu: ca.MX) -> ca.MX:
        quad = ca.fabs(nu) * nu  # |nu|*nu elementwise
        return (self.DL @ nu) + (self.DQ @ quad)

    def _f_cont_expr(self, x: ca.MX, tau: ca.MX) -> ca.MX:
        _ = x[0:3]
        quat = x[3:7]
        nu = x[7:13]

        v_b = nu[0:3]
        omega_b = nu[3:6]

        R = quat_to_rotmat_ca(quat)

        pos_dot = R @ v_b
        quat_dot = omega_to_quat_rate_ca(quat, omega_b)

        C = coriolis_from_mass_ca(self.MRB, nu) + coriolis_from_mass_ca(self.MA, nu)
        gq = self._restoring(quat)
        dnu = self._damping(nu)

        rhs = tau - (C @ nu) - dnu - gq
        nu_dot = self.Minv @ rhs

        return ca.vertcat(pos_dot, quat_dot, nu_dot)

    def _rk4_step_expr(self, x: ca.MX, u: ca.MX) -> ca.MX:
        dt = self.dt

        k1 = self._f_cont_expr(x, u)
        k2 = self._f_cont_expr(x + 0.5 * dt * k1, u)
        k3 = self._f_cont_expr(x + 0.5 * dt * k2, u)
        k4 = self._f_cont_expr(x + dt * k3, u)

        x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Quaternion normalization (smooth)
        q_next = x_next[3:7]
        qn = ca.sqrt(ca.dot(q_next, q_next) + self.quat_norm_eps)
        q_next = q_next / qn
        x_next = ca.vertcat(x_next[0:3], q_next, x_next[7:13])

        return x_next

    def _build_functions(self) -> None:
        x = ca.MX.sym("x", self.nx)
        u = ca.MX.sym("u", self.nu)

        xdot = self._f_cont_expr(x, u)
        xnext = self._rk4_step_expr(x, u)

        self.f_cont = ca.Function("f_cont", [x, u], [xdot], ["x", "u"], ["xdot"])
        self.f_disc = ca.Function("f_disc", [x, u], [xnext], ["x", "u"], ["x_next"])

    def step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Numeric step using the CasADi discrete function."""
        x = np.asarray(x, float).reshape(self.nx)
        u = np.asarray(u, float).reshape(self.nu)

        # Normalize input quaternion for safety
        x = x.copy()
        x[3:7] = quat_normalize_np(x[3:7])

        xn = self.f_disc(x, u)
        return np.array(xn).reshape(self.nx)


# ----------------------------
# NMPC (Direct multiple shooting)
# ----------------------------
class AuvNMPC:
    """
    NMPC for x=[pos(3), quat(4), nu(6)] and u=tau(6).

    Parameters passed at solve-time:
      - x0 (13,)
      - x_ref trajectory (13, N+1) OR a single (13,) setpoint (broadcasted)
    """

    def __init__(
        self,
        model: UnderwaterVehicle6DOF_CasADi,
        N: int,
        # weights
        Q_pos: np.ndarray,
        Q_vel: np.ndarray,
        Q_ori: float,
        R_tau: np.ndarray,
        QN_pos: Optional[np.ndarray] = None,
        QN_vel: Optional[np.ndarray] = None,
        QN_ori: Optional[float] = None,
        # bounds
        tau_min: Optional[np.ndarray] = None,
        tau_max: Optional[np.ndarray] = None,
        nu_min: Optional[np.ndarray] = None,
        nu_max: Optional[np.ndarray] = None,
        # solver opts
        ipopt_opts: Optional[Dict[str, Any]] = None,
        compiled_solver_path: Optional[str] = None,
    ):
        self.model = model
        self.N = int(N)

        self.nx = model.nx
        self.nu = model.nu

        # Store default diagonals (used when solve() is called without overriding weights)
        Q_pos = np.asarray(Q_pos, float).reshape(3, 3)
        Q_vel = np.asarray(Q_vel, float).reshape(6, 6)
        R_tau = np.asarray(R_tau, float).reshape(6, 6)

        if QN_pos is None:
            QN_pos = Q_pos
        if QN_vel is None:
            QN_vel = Q_vel
        if QN_ori is None:
            QN_ori = Q_ori

        QN_pos = np.asarray(QN_pos, float).reshape(3, 3)
        QN_vel = np.asarray(QN_vel, float).reshape(6, 6)

        self.q_pos0 = np.diag(Q_pos).copy()  # (3,)
        self.q_vel0 = np.diag(Q_vel).copy()  # (6,)
        self.r_tau0 = np.diag(R_tau).copy()  # (6,)
        self.qN_pos0 = np.diag(QN_pos).copy()  # (3,)
        self.qN_vel0 = np.diag(QN_vel).copy()  # (6,)
        self.q_ori0 = float(Q_ori)
        self.qN_ori0 = float(QN_ori)

        self.tau_min = (
            None if tau_min is None else np.asarray(tau_min, float).reshape(6)
        )
        self.tau_max = (
            None if tau_max is None else np.asarray(tau_max, float).reshape(6)
        )
        self.nu_min = None if nu_min is None else np.asarray(nu_min, float).reshape(6)
        self.nu_max = None if nu_max is None else np.asarray(nu_max, float).reshape(6)

        self._build_solver(
            ipopt_opts=ipopt_opts, compiled_solver_path=compiled_solver_path
        )

        # Warm start storage
        self._w_last = None
        self._lam_x_last = None
        self._lam_g_last = None

    def _stage_cost_diag(
        self,
        xk: ca.MX,
        uk: ca.MX,
        xrefk: ca.MX,
        q_pos: ca.MX,  # (3,)
        q_vel: ca.MX,  # (6,)
        r_tau: ca.MX,  # (6,)
        q_ori: ca.MX,  # scalar
    ) -> ca.MX:
        ep = xk[0:3] - xrefk[0:3]
        ev = xk[7:13] - xrefk[7:13]

        # diagonal quadratic: sum_i w_i * e_i^2
        Jp = ca.dot(q_pos, ep * ep)
        Jv = ca.dot(q_vel, ev * ev)
        Ju = ca.dot(r_tau, uk * uk)

        q = xk[3:7]
        qref = xrefk[3:7]
        qerr = quat_mult_ca(quat_conj_ca(qref), q)
        Jq = q_ori * ca.dot(qerr[0:3], qerr[0:3])

        return Jp + Jv + Jq + Ju

    def _terminal_cost_diag(
        self,
        xN: ca.MX,
        xrefN: ca.MX,
        qN_pos: ca.MX,  # (3,)
        qN_vel: ca.MX,  # (6,)
        qN_ori: ca.MX,  # scalar
    ) -> ca.MX:
        ep = xN[0:3] - xrefN[0:3]
        ev = xN[7:13] - xrefN[7:13]

        Jp = ca.dot(qN_pos, ep * ep)
        Jv = ca.dot(qN_vel, ev * ev)

        q = xN[3:7]
        qref = xrefN[3:7]
        qerr = quat_mult_ca(quat_conj_ca(qref), q)
        Jq = qN_ori * ca.dot(qerr[0:3], qerr[0:3])

        return Jp + Jv + Jq

    def _build_solver(
        self,
        ipopt_opts: Optional[Dict[str, Any]],
        compiled_solver_path: Optional[str] = None,
    ) -> None:
        N = self.N
        nx, nu = self.nx, self.nu

        # Solver options
        opts = {
            "print_time": True,
            "ipopt.print_level": 1,
            "ipopt.sb": "yes",
            "ipopt.max_iter": 100,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-2,
            "expand": True,
        }
        if ipopt_opts:
            opts.update(ipopt_opts)

        if compiled_solver_path:
            # When loading a compiled solver, "expand" is typically not relevant or used
            if "expand" in opts:
                opts.pop("expand")

            self.solver = ca.nlpsol("solver", "ipopt", compiled_solver_path, opts)

            # Reconstruct constraint bounds size
            # g structure: [x0_constraint(nx), N dynamics constraints(nx*N)]
            # Total N+1 blocks of size nx.
            ng = nx * (N + 1)
            self.lbg = np.zeros(ng)
            self.ubg = np.zeros(ng)

            # Variable size
            nW = nx * (N + 1) + nu * N
            self._U_offset = nx * (N + 1)
            self._n_weights = 26
            self._p_size = nx + nx * (N + 1) + self._n_weights
        else:
            # Decision variables
            X = ca.MX.sym("X", nx, N + 1)
            U = ca.MX.sym("U", nu, N)

            # Parameters: x0, x_ref trajectory, and weights
            X0 = ca.MX.sym("X0", nx)
            XREF = ca.MX.sym("XREF", nx, N + 1)

            # Weight parameters (diagonals + orientation scalars)
            QPOS = ca.MX.sym("QPOS", 3)  # position diag
            QVEL = ca.MX.sym("QVEL", 6)  # nu diag
            RTAU = ca.MX.sym("RTAU", 6)  # control diag
            QNPOS = ca.MX.sym("QNPOS", 3)  # terminal position diag
            QNVEL = ca.MX.sym("QNVEL", 6)  # terminal nu diag
            QORI = ca.MX.sym("QORI", 1)  # scalar
            QNORI = ca.MX.sym("QNORI", 1)  # scalar

            g = []
            obj = 0

            # Initial condition constraint
            g.append(X[:, 0] - X0)

            # Dynamics and cost
            for k in range(N):
                xk = X[:, k]
                uk = U[:, k]
                x_next = self.model.f_disc(xk, uk)
                g.append(X[:, k + 1] - x_next)

                obj = obj + self._stage_cost_diag(
                    xk, uk, XREF[:, k], QPOS, QVEL, RTAU, QORI
                )

            obj = obj + self._terminal_cost_diag(
                X[:, N], XREF[:, N], QNPOS, QNVEL, QNORI
            )

            g = ca.vertcat(*g)

            w = ca.vertcat(ca.reshape(X, nx * (N + 1), 1), ca.reshape(U, nu * N, 1))

            p = ca.vertcat(
                X0,
                ca.reshape(XREF, nx * (N + 1), 1),
                QPOS,
                QVEL,
                RTAU,
                QNPOS,
                QNVEL,
                QORI,
                QNORI,
            )

            self._n_weights = 3 + 6 + 6 + 3 + 6 + 1 + 1  # = 26
            self._p_size = nx + nx * (N + 1) + self._n_weights

            nlp = {"x": w, "p": p, "f": obj, "g": g}

            self.solver = ca.nlpsol("solver", "ipopt", nlp, opts)

            # Constraint bounds (all equalities)
            ng = int(g.size1())
            self.lbg = np.zeros(ng)
            self.ubg = np.zeros(ng)

            # Variable size
            nW = int(w.size1())
            self._U_offset = nx * (N + 1)

        # Variable bounds (common logic)
        lbx = -np.inf * np.ones(nW)
        ubx = np.inf * np.ones(nW)

        # Bounds for quaternion components (optional but often stabilizes)
        # Flattened X is column-major: index = k*nx + i
        for k in range(N + 1):
            base = k * nx
            for i in range(3, 7):
                lbx[base + i] = -1.2
                ubx[base + i] = 1.2

        # Velocity bounds (nu)
        if self.nu_min is not None:
            for k in range(N + 1):
                base = k * nx
                lbx[base + 7 : base + 13] = self.nu_min
        if self.nu_max is not None:
            for k in range(N + 1):
                base = k * nx
                ubx[base + 7 : base + 13] = self.nu_max

        # Control bounds (tau)
        U_offset = self._U_offset
        if self.tau_min is not None:
            for k in range(N):
                base = U_offset + k * nu
                lbx[base : base + nu] = self.tau_min
        if self.tau_max is not None:
            for k in range(N):
                base = U_offset + k * nu
                ubx[base : base + nu] = self.tau_max

        self.lbx = lbx
        self.ubx = ubx

    def generate_c_code(self, filename: str = "mpc_solver.c") -> None:
        """
        Export the solver as C source code for compilation.

        Example compilation command:
            gcc -fPIC -shared -O3 mpc_solver.c -o mpc_solver.so
        """
        self.solver.generate_dependencies(filename)
        print(f"Solver C code generated at: {filename}")

    def solve(
        self,
        x0: np.ndarray,
        x_ref: np.ndarray,
        w0: Optional[np.ndarray] = None,
        weights: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve NMPC.

        x0: (13,)
        x_ref: either (13,) setpoint OR (13, N+1) reference trajectory
        weights: optional dict with keys "q_pos", "q_vel", "r_tau", "qN_pos", "qN_vel", "q_ori", "qN_ori"

        Returns:
          u0: (6,) first control
          X_pred: (13, N+1) predicted trajectory
          U_pred: (6, N) predicted controls
        """
        nx, nu, N = self.nx, self.nu, self.N

        x0 = np.asarray(x0, float).reshape(nx).copy()
        x0[3:7] = quat_normalize_np(x0[3:7])

        x_ref = np.asarray(x_ref, float)
        if x_ref.ndim == 1:
            x_ref = x_ref.reshape(nx, 1)
            x_ref = np.repeat(x_ref, N + 1, axis=1)
        else:
            x_ref = x_ref.reshape(nx, N + 1).copy()
            # normalize reference quaternions
            for k in range(N + 1):
                x_ref[3:7, k] = quat_normalize_np(x_ref[3:7, k])

        # --- weights (runtime-tunable) ---
        w = weights or {}
        q_pos = np.asarray(w.get("q_pos", self.q_pos0), float).reshape(3)
        q_vel = np.asarray(w.get("q_vel", self.q_vel0), float).reshape(6)
        r_tau = np.asarray(w.get("r_tau", self.r_tau0), float).reshape(6)
        qN_pos = np.asarray(w.get("qN_pos", self.qN_pos0), float).reshape(3)
        qN_vel = np.asarray(w.get("qN_vel", self.qN_vel0), float).reshape(6)
        q_ori = float(w.get("q_ori", self.q_ori0))
        qN_ori = float(w.get("qN_ori", self.qN_ori0))

        # Safety: keep weights non-negative
        eps = 1e-12
        q_pos = np.maximum(q_pos, eps)
        q_vel = np.maximum(q_vel, eps)
        r_tau = np.maximum(r_tau, eps)
        qN_pos = np.maximum(qN_pos, eps)
        qN_vel = np.maximum(qN_vel, eps)
        q_ori = max(q_ori, eps)
        qN_ori = max(qN_ori, eps)

        # Pack parameters
        p = np.concatenate(
            [
                x0,
                x_ref.reshape(nx * (N + 1), order="F"),
                q_pos,
                q_vel,
                r_tau,
                qN_pos,
                qN_vel,
                np.array([q_ori, qN_ori], dtype=float),
            ]
        )

        # Initial guess
        if w0 is None:
            if self._w_last is not None:
                w0 = self._shift_guess(self._w_last)
            else:
                w0 = np.zeros(self.nx * (N + 1) + self.nu * N)
                # Seed initial state guess
                for k in range(N + 1):
                    w0[k * nx : (k + 1) * nx] = x0
        else:
            w0 = np.asarray(w0, float).reshape(-1)

        # Solve
        arg = dict(x0=w0, p=p, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg)

        # Warm-start multipliers if available
        if self._lam_x_last is not None and self._lam_g_last is not None:
            arg["lam_x0"] = self._lam_x_last
            arg["lam_g0"] = self._lam_g_last

        sol = self.solver(**arg)

        w_opt = np.array(sol["x"]).reshape(-1)
        self._w_last = w_opt
        self._lam_x_last = sol.get("lam_x", None)
        self._lam_g_last = sol.get("lam_g", None)

        X_flat = w_opt[: self.nx * (N + 1)]
        U_flat = w_opt[self.nx * (N + 1) :]

        X_pred = X_flat.reshape((nx, N + 1), order="F")
        U_pred = U_flat.reshape((nu, N), order="F")

        u0 = U_pred[:, 0].copy()
        return u0, X_pred, U_pred

    def _shift_guess(self, w: np.ndarray) -> np.ndarray:
        """Shift previous solution forward by one step for warm-start."""
        nx, nu, N = self.nx, self.nu, self.N

        X_flat = w[: nx * (N + 1)]
        U_flat = w[nx * (N + 1) :]
        X = X_flat.reshape((nx, N + 1), order="F")
        U = U_flat.reshape((nu, N), order="F")

        Xs = np.zeros_like(X)
        Us = np.zeros_like(U)

        Xs[:, 0:N] = X[:, 1 : N + 1]
        Xs[:, N] = X[:, N]  # keep last
        Us[:, 0 : N - 1] = U[:, 1:N]
        Us[:, N - 1] = U[:, N - 1]  # keep last

        return np.concatenate(
            [Xs.reshape(nx * (N + 1), order="F"), Us.reshape(nu * N, order="F")]
        )
