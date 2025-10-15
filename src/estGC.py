import numpy as np
from numpy.linalg import lstsq


def compute_dTdt(T_nodes: np.ndarray, dt: float) -> np.ndarray:
    """
    Centered finite difference for each column (node).
    T_nodes: (N, 2) -> columns: [T_s, T_r]
    returns dTdt shape (N, 2)
    """
    N, n = T_nodes.shape
    dTdt = np.zeros_like(T_nodes)
    if N < 2:
        raise ValueError("Need at least 2 time points")
    # centered differences for interior
    if N > 2:
        dTdt[1:-1] = (T_nodes[2:] - T_nodes[:-2]) / (2.0 * dt)
    # forward/backward for edges
    dTdt[0] = (T_nodes[1] - T_nodes[0]) / dt
    dTdt[-1] = (T_nodes[-1] - T_nodes[-2]) / dt
    return dTdt


def est_CG(T_nodes: np.ndarray, P_inputs: np.ndarray, Tamb: np.ndarray, dt: float):
    """
    Estimate C_s, C_r, g_sr, g_sa, g_ra from time-series data.
    Inputs:
      T_nodes : (N,2)  -> [T_s, T_r]
      P_inputs: (N,2)  -> [P_s, P_r] (W)
      Tamb    : (N,)   -> ambient temperature (°C)
      dt      : scalar time step (s)
    Returns:
      C_diag : array([C_s, C_r])
      G_mat  : 2x2 conductance matrix G (watts/K)
      g_vals : dict with keys 'g_sr','g_sa','g_ra'
      residuals_norm : norm of LS residual (for diagnostics)
    """
    N, n = T_nodes.shape
    assert n == 2
    assert P_inputs.shape == (N, 2)
    assert Tamb.shape[0] == N

    dTdt = compute_dTdt(T_nodes, dt)  # (N,2)

    # Build linear system: for each time step i we have two equations:
    # Eq_s: C_s * dT_s   + g_sr*(T_s - T_r) + g_sa*(T_s - T_amb) = P_s
    # Eq_r: C_r * dT_r   + g_sr*(T_r - T_s) + g_ra*(T_r - T_amb) = P_r
    #
    # Unknown vector theta = [C_s, C_r, g_sr, g_sa, g_ra] (5 unknowns)
    # For each time step we build row_s and row_r (length 5), and stack.

    rows = []
    targets = []

    T_s = T_nodes[:, 0]
    T_r = T_nodes[:, 1]
    P_s = P_inputs[:, 0]
    P_r = P_inputs[:, 1]

    for i in range(N):
        # equation for stator (row_s)
        # Coeffs multiply unknowns [C_s, C_r, g_sr, g_sa, g_ra]
        row_s = np.array([
            dTdt[i, 0],   # multiplies C_s
            0.0,          # C_r not present in eq_s
            (T_s[i] - T_r[i]),  # multiplies g_sr
            (T_s[i] - Tamb[i]), # multiplies g_sa
            0.0           # g_ra not present in eq_s
        ], dtype=float)
        rows.append(row_s)
        targets.append(P_s[i])

        # equation for rotor (row_r)
        row_r = np.array([
            0.0,                # C_s not present in eq_r
            dTdt[i, 1],         # multiplies C_r
            (T_r[i] - T_s[i]),  # multiplies g_sr (note sign)
            0.0,
            (T_r[i] - Tamb[i])  # multiplies g_ra
        ], dtype=float)
        rows.append(row_r)
        targets.append(P_r[i])

    Phi = np.vstack(rows)        # (2N, 5)
    y = np.vstack(targets).ravel()  # (2N,)

    # Solve least squares
    theta, *_ = lstsq(Phi, y, rcond=None)
    C_s, C_r, g_sr, g_sa, g_ra = theta

    # Build matrices
    C_diag = np.array([C_s, C_r])
    # Conductance matrix G (positive diag, negative off-diag)
    G = np.array([
        [g_sa + g_sr, -g_sr],
        [-g_sr,       g_ra + g_sr]
    ])

    # Diagnostics
    residuals = Phi @ theta - y
    res_norm = np.linalg.norm(residuals) / np.sqrt(len(y))

    g_vals = {"g_sr": g_sr, "g_sa": g_sa, "g_ra": g_ra}

    return C_diag, G, g_vals, res_norm


def calc_T(C_diag, G, P_inputs, Tamb, T0=None, dt=1.0):
    """
    Forward Euler simulation: C dT/dt = -G (T - Tamb) + P  => dT/dt = C^{-1} ( -G(T - Tamb) + P )
    C_diag: [C_s, C_r]
    G: 2x2 conductance matrix
    P_inputs: (N,2)
    Tamb: (N,)
    """
    N = P_inputs.shape[0]
    invC = np.diag(1.0 / C_diag)
    T = np.zeros((N, 2))
    if T0 is None:
        T[0] = Tamb[0]  # a reasonable init
    else:
        T[0] = T0

    for i in range(1, N):
        rhs = - G @ (T[i-1] - np.array([Tamb[i-1], Tamb[i-1]])) + P_inputs[i-1]
        dT = invC @ rhs
        T[i] = T[i-1] + dt * dT
    return T
