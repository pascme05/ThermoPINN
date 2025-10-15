import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


# ----------------------------------------------------
# Helper functions
# ----------------------------------------------------
def identify_rc(P: np.ndarray, T: np.ndarray, Tamb: np.ndarray, dt: float) -> tuple[float, float]:
    """
    Identify thermal resistance R and capacitance C from step response.

    Parameters
    ----------
    P : np.ndarray
        Power input array (W)
    T : np.ndarray
        Temperature array (°C)
    Tamb : np.ndarray
        Ambient temperature array (°C)
    dt : float
        Time step (s)

    Returns
    -------
    R : float
        Thermal resistance (K/W)
    C : float
        Thermal capacitance (J/K)
    """
    dTdt = (T[1:] - T[:-1]) / dt
    T_mid = T[:-1]
    T_amb = Tamb[:-1]
    P_mid = P[:-1]

    Phi = np.column_stack([P_mid, -(T_mid - T_amb)])
    theta, *_ = np.linalg.lstsq(Phi, dTdt, rcond=None)
    invC, invRC = theta
    C = 1.0 / invC
    R = 1.0 / (invRC * C)
    return R, C


def foster_rc(T_amb: np.ndarray, P: np.ndarray, dt: float, Rth: float, Cth: float) -> np.ndarray:
    """
    Simulate a 1-node Foster RC thermal model.

    Parameters
    ----------
    T_amb : np.ndarray
        Ambient temperature (°C)
    P : np.ndarray
        Input power (W)
    dt : float
        Time step (s)
    Rth : float
        Thermal resistance (K/W)
    Cth : float
        Thermal capacitance (J/K)

    Returns
    -------
    T_est : np.ndarray
        Estimated junction temperature
    """
    Nt = len(P)
    tau = Rth * Cth
    T_est = np.zeros(Nt)
    alpha = (2 * tau - dt) / (2 * tau + dt)
    beta = (Rth * dt) / (2 * tau + dt)

    for i in range(1, Nt):
        T_est[i] = alpha * T_est[i - 1] + beta * (P[i] + P[i - 1])

    return T_est + T_amb


def make_sequences(X: np.ndarray, T: np.ndarray, P: np.ndarray, Tamb: np.ndarray,
                   t: np.ndarray, T0: np.ndarray, seq_len: int, stride: int = 1) -> tuple[np.ndarray, ...]:
    """
    Convert time-series data into overlapping sequences.

    Returns
    -------
    Tuple of np.ndarray: X_seq, T_seq, P_seq, Tamb_seq, t_seq, T0_seq
    """
    X_seq, T_seq, P_seq, Tamb_seq, t_seq, T0_seq = [], [], [], [], [], []
    for i in range(0, len(X) - seq_len + 1, stride):
        X_seq.append(X[i:i + seq_len])
        T_seq.append(T[i:i + seq_len])
        P_seq.append(P[i:i + seq_len])
        Tamb_seq.append(Tamb[i:i + seq_len])
        t_seq.append(t[i:i + seq_len])
        T0_seq.append(T0[i:i + seq_len])
    return (np.array(X_seq), np.array(T_seq), np.array(P_seq),
            np.array(Tamb_seq), np.array(t_seq), np.array(T0_seq))


def normalize(df_part: pd.DataFrame, feature_cols: list[str], X_mean: pd.Series, X_std: pd.Series,
              T_max: float, T_min: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features and target temperature for neural network input.
    """
    X = (df_part[feature_cols] - X_mean) / X_std
    T = (df_part["Tsw"].values - T_min) / (T_max - T_min)
    Tamb = (df_part["Tc"].values - T_min) / (T_max - T_min)
    return X.values, T, Tamb


def prepare_loader(X: np.ndarray, T: np.ndarray, P: np.ndarray, Tamb: np.ndarray,
                   t: np.ndarray, T0: np.ndarray, seq_len: int, stride: int,
                   batch_size: int, device: str, shuffle: bool) -> DataLoader:
    """
    Convert sequences to PyTorch DataLoader.
    """
    X_seq, T_seq, P_seq, Tamb_seq, t_seq, T0_seq = make_sequences(X, T, P, Tamb, t, T0, seq_len, stride)
    X_seq = torch.tensor(X_seq, dtype=torch.float32, device=device)
    T_seq = torch.tensor(T_seq, dtype=torch.float32, device=device)
    P_seq = torch.tensor(P_seq, dtype=torch.float32, device=device)
    Tamb_seq = torch.tensor(Tamb_seq, dtype=torch.float32, device=device)
    t_seq = torch.tensor(t_seq, dtype=torch.float32, device=device)
    T0_seq = torch.tensor(T0_seq, dtype=torch.float32, device=device)
    dataset = TensorDataset(X_seq, T_seq, P_seq, Tamb_seq, t_seq, T0_seq)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def gradient(T: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
    """
    Compute dT/dt using centered finite differences.
    """
    dT = torch.zeros_like(T)
    dT[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / (2 * dt)
    dT[:, 0] = (T[:, 1] - T[:, 0]) / dt
    dT[:, -1] = (T[:, -1] - T[:, -2]) / dt
    return dT
