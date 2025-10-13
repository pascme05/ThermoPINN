# ----------------------------------------------------
# pinn_thermal_lstm_csv.py
# Author: Pascal Schirmer
# Date: 2025-10-12
# Description:
#   Physics-informed neural network (PINN) using an LSTM model
#   for thermal prediction from CSV input data. The script
#   performs RC identification, model training, and evaluation
#   with optional visualization.
# ----------------------------------------------------

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


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
              T_max: float, T_min: float, Rs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features and target temperature for neural network input.
    """
    X = (df_part[feature_cols] - X_mean) / X_std
    T = (df_part["Tsw"].values - T_min) / (T_max - T_min)
    P = 3 * Rs * df_part["Is"].values ** 2
    Tamb = (df_part["Tc"].values - T_min) / (T_max - T_min)
    return X.values, T, P, Tamb


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


def pinn_loss_lstm(model: nn.Module, X: torch.Tensor, T: torch.Tensor, P: torch.Tensor,
                   t: torch.Tensor, T0: torch.Tensor, dt: torch.Tensor, R: float, C: float,
                   Tamb: torch.Tensor, Tmin: float, Tmax: float, lambda_phys: float = 1.0, lambda_init: float = 1.0) \
                    -> tuple[torch.Tensor, float, float, float]:
    """
    Physics-informed loss for LSTM-PINN: data + physics + initial condition.
    """
    T_pred = model(X)
    dTdt_pred = gradient(T_pred, dt) * (Tmax - Tmin)

    T_t = T_pred * (Tmax - Tmin) + Tmin
    Tamb_phys = Tamb * (Tmax - Tmin) + Tmin

    rhs = (1.0 / C) * P - (1.0 / (R * C)) * (T_t - Tamb_phys)
    res = dTdt_pred - rhs

    weights = torch.exp(-t / np.sqrt(R * C)).unsqueeze(0)
    ic_mse = torch.mean(weights * ((T_t - T0) / (Tmax - Tmin)) ** 2)

    data_mse = torch.mean((T_pred - T) ** 2)
    phys_mse = torch.mean(res ** 2)
    total = data_mse + lambda_phys * phys_mse + lambda_init * ic_mse
    return total, data_mse.item(), lambda_phys * phys_mse.item(), lambda_init * ic_mse.item()


# ----------------------------------------------------
# LSTM-PINN Model
# ----------------------------------------------------
class LSTM_PINN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out.squeeze(-1)


# ----------------------------------------------------
# Main workflow
# ----------------------------------------------------
def main():
    # -------------------------------
    # Config / Parameters
    # -------------------------------
    TRAIN_MODEL = True
    ENABLE_PLOTS = True

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "motor_temp.csv")

    # Thermal / motor parameters
    Rs = 14.1e-3                                                                                                         # Stator resistance [Ohm]
    alpha = 0.00393                                                                                                      # Temperature coefficient of resistance [1/°C]
    Tref = 20                                                                                                            # Reference temperature for Rs [°C]
    n_max = 6000                                                                                                         # Maximum motor speed [rpm]
    beta_1 = 0.315                                                                                                       # Parameter for frequency losses (linear)
    beta_2 = 0.616                                                                                                       # Parameter for frequency losses (quadratic)

    # Training hyperparameters
    seq_len = 600                                                                                                        # Sequence length (timesteps per training sample)
    stride = 50                                                                                                           # Step size between training sequences
    batch_size = 32                                                                                                      # Batch size for training
    hidden_dim = 8                                                                                                      # Hidden units in LSTM layers
    num_layers = 2                                                                                                       # Number of stacked LSTM layers
    lr = 1e-3                                                                                                            # Learning rate for optimizer
    epochs = 20                                                                                                         # Maximum number of training epochs
    lambda_phys = 0.1                                                                                                    # Weight for physics-informed loss term
    lambda_init = 0.0                                                                                                    # Weight for initial condition loss (currently unused)
    patience = 10                                                                                                        # Early stopping patience (epochs without improvement)

    # Dataset split IDs
    test_ids = [60, 62, 74]                                                                                              # IDs used for test set evaluation
    val_ids = [10, 48, 63]                                                                                               # IDs used for validation set selection

    # -------------------------------
    # Load CSV data
    # -------------------------------
    df = pd.read_csv(DATA_PATH)

    # -------------------------------
    # RC Identification
    # -------------------------------
    id_list = [2, 3, 5, 7, 8, 9, 12, 13, 14, 15, 16, 17, 19, 21, 24]
    id_data = []
    dt_s = 1

    for id_sel in id_list:
        df_step = df[df["id"] == id_sel].copy().head(3500)
        time_step = df_step["time"].values - df_step["time"].values[0]
        T_amb = df_step["Tc"].values
        T_step = df_step["Tsw"].values
        Is = df_step["Is"].values
        Wm = df_step["Wm"].values

        f1 = (1 + alpha * (T_step - Tref))
        f2 = 1 + beta_1 * (Wm / n_max) + beta_2 * (Wm / n_max) ** 2
        P_step = 3 * Rs * Is ** 2 * f1 * f2
        dt_s = np.mean(np.diff(time_step))

        R_fit, C_fit = identify_rc(P_step.flatten(), T_step, T_amb, dt_s)
        id_data.append({"id": id_sel, "Is": np.mean(Is), "Wm": np.mean(Wm),
                        "Pv": np.max(P_step), "R": R_fit, "C": C_fit})

    df_ident = pd.DataFrame(id_data)

    # Polynomial fit (optional, can be used later)
    poly_R = make_pipeline(PolynomialFeatures(2), LinearRegression())
    poly_R.fit(df_ident[["Is", "Wm"]], df_ident["R"])
    poly_C = make_pipeline(PolynomialFeatures(2), LinearRegression())
    poly_C.fit(df_ident[["Is", "Wm"]], df_ident["C"])

    R_hat = df_ident["R"].mean()
    C_hat = df_ident["C"].mean()
    print(f"Identified Average: R = {R_hat:.4f} K/W, C = {C_hat:.2f} J/K")

    # -------------------------------
    # Feature preparation
    # -------------------------------
    # Calc features
    Pv1 = 3 * Rs * df["Is"] ** 2
    T_rc = foster_rc(df["Tc"], Pv1, dt_s, R_hat, C_hat)
    f1 = (1 + alpha * (T_rc - Tref))
    f2 = 1 + beta_1 * (df["Wm"] / n_max) + beta_2 * (df["Wm"] / n_max) ** 2
    df["Pv_s"] = 3 * Rs * df["Is"] ** 2 * f1 * f2
    df["Sel"] = 3 * df["Is"] * df["Us"]
    df["SelI"] = df["Sel"] * df["Is"]
    df["SelW"] = df["Sel"] * df["Wm"]

    # Remove features
    feature_cols = [c for c in df.columns if c not in ["id", "time", "time_id", "T0", "Tsw", "Tst", "Tso", "Trm"]]

    # -------------------------------
    # Data Splitting
    # -------------------------------
    train_ids = [i for i in df["id"].unique() if i not in test_ids + val_ids]
    df_train = df[df["id"].isin(train_ids)].copy()
    df_val = df[df["id"].isin(val_ids)].copy()
    df_test = df[df["id"].isin(test_ids)].copy()

    # -------------------------------
    # Normalize and Scale
    # -------------------------------
    # Norm values
    X_mean, X_std = df_train[feature_cols].mean(), df_train[feature_cols].std() + 1e-8
    T_min, T_max = df_train["Tsw"].min(), df_train["Tsw"].max()

    # Normalize
    X_train, T_train, P_train, Tamb_train = normalize(df_train, feature_cols, X_mean, X_std, T_max, T_min, Rs)
    X_val, T_val, P_val, Tamb_val = normalize(df_val, feature_cols, X_mean, X_std, T_max, T_min, Rs)
    # X_test, T_test, P_test, Tamb_test = normalize(df_test, feature_cols, X_mean, X_std, T_max, T_min, Rs)

    # Scale Temperature-dependent Power
    P_train *= (1 + alpha * (df_train["Tsw"] - Tref)) * (1 + beta_1 * (df_train["Wm"] / n_max) + beta_2 * (df_train["Wm"] / n_max) ** 2)
    P_val *= (1 + alpha * (df_val["Tsw"] - Tref)) * (1 + beta_1 * (df_val["Wm"] / n_max) + beta_2 * (df_val["Wm"] / n_max) ** 2)

    # -------------------------------
    # DataLoaders
    # -------------------------------
    train_loader = prepare_loader(X_train, T_train, P_train, Tamb_train, df_train["time_id"].values,
                                  df_train["T0"].values, seq_len, stride, batch_size, DEVICE, shuffle=False)
    val_loader = prepare_loader(X_val, T_val, P_val, Tamb_val, df_val["time_id"].values,
                                df_val["T0"].values, seq_len, stride, batch_size, DEVICE, shuffle=False)

    # -------------------------------
    # Model setup
    # -------------------------------
    n_features = len(feature_cols)
    model = LSTM_PINN(input_dim=n_features, hidden_dim=hidden_dim, num_layers=num_layers).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    dt_torch = torch.tensor(dt_s, dtype=torch.float32, device=DEVICE)

    best_val_loss = np.inf
    patience_counter = 0

    # -------------------------------
    # Training loop
    # -------------------------------
    if TRAIN_MODEL:
        for ep in range(epochs):
            model.train()
            total_loss, data_loss, phys_loss, init_loss = 0, 0, 0, 0

            for Xb, Tb, Pb, Tambb, t_seq, T0b in train_loader:
                optimizer.zero_grad()
                loss, d_mse, p_mse, i_mse = pinn_loss_lstm(model, Xb, Tb, Pb, t_seq, T0b, dt_torch,
                                                           R_hat, C_hat, Tambb, T_min, T_max,
                                                           lambda_phys, lambda_init)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                data_loss += d_mse
                phys_loss += p_mse
                init_loss += i_mse

            # Validation
            model.eval()
            val_total, val_data, val_phys, val_init = 0, 0, 0, 0
            with torch.no_grad():
                for Xv, Tv, Pv, Tambv, tv_seq, T0v in val_loader:
                    v_loss, v_dmse, v_pmse, v_imse = pinn_loss_lstm(model, Xv, Tv, Pv, tv_seq, T0v, dt_torch,
                                                                    R_hat, C_hat, Tambv, T_min, T_max,
                                                                    lambda_phys, lambda_init)
                    val_total += v_loss.item()
                    val_data += v_dmse
                    val_phys += v_pmse
                    val_init += v_imse

            train_loss = total_loss / len(train_loader)
            val_loss = val_total / len(val_loader)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {ep+1:03d} | LR={current_lr:.6f} | "
                  f"Train={train_loss:.6f} | Val={val_loss:.6f} | "
                  f"Data={val_data/len(val_loader):.6f} | Phys={val_phys/len(val_loader):.6f} | "
                  f"Init={val_init/len(val_loader):.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "mdl/mdl_best_pinn.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    # -------------------------------
    # Test evaluation
    # -------------------------------
    model.load_state_dict(torch.load("mdl/mdl_best_pinn.pt"))
    model.eval()

    for sid in test_ids:
        print(f"\n🔹 Evaluating Test Session ID: {sid}")

        # Select session-specific data
        df_session = df_test[df_test["id"] == sid].copy()
        if df_session.empty:
            print(f"⚠️ No data found for session {sid}, skipping.")
            continue

        # Prepare physics-informed scaling
        P_test = df_session["Pv_s"].values

        # RC model prediction
        T_pred_rc = foster_rc(df_session["Tc"].values, P_test, dt_s, R_hat, C_hat)

        # Prepare neural network input
        X_test, T_test, _, _ = normalize(df_session, feature_cols, X_mean, X_std, T_max, T_min, Rs)

        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            T_pred_nn = model(X_tensor).cpu().numpy().reshape(-1)

            # Inverse normalization
            T_true_nn = T_test * (T_max - T_min) + T_min
            T_pred_nn_phys = T_pred_nn * (T_max - T_min) + T_min

            # Compute errors
            err_rc = T_pred_rc - T_true_nn
            err_nn = T_pred_nn_phys - T_true_nn

            mse_test_rc = np.mean(err_rc ** 2)
            mse_test_nn = np.mean(err_nn ** 2)

            print(f"RC MSE (°C): {mse_test_rc:.4f}")
            print(f"NN MSE (°C): {mse_test_nn:.4f}")

            # -------------------------------
            # Plotting per session
            # -------------------------------
            if ENABLE_PLOTS:
                time = np.linspace(0, (len(err_rc) - 1) / 60, len(err_rc))  # time in minutes
                fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                fig.suptitle(f"Session ID {sid} – Model Comparison", fontsize=12, fontweight="bold")

                # ▪️ Top: Temperatures
                axs[0].plot(time, T_true_nn, label="Measured", color="black", linewidth=2)
                axs[0].plot(time, T_pred_rc, label="Pred RC", linewidth=1.8)
                axs[0].plot(time, T_pred_nn_phys, label="Pred NN", linewidth=1.8)
                axs[0].set_ylabel("Temperature [°C]")
                axs[0].legend(loc="best")
                axs[0].grid(True, linestyle="--", linewidth=0.6)

                # ▪️ Bottom: Errors
                axs[1].plot(time, err_rc, label="RC Error")
                axs[1].plot(time, err_nn, label="NN Error")
                axs[1].axhline(0, color="black", linewidth=1)
                axs[1].set_xlabel("Time [min]")
                axs[1].set_ylabel("Error [°C]")
                axs[1].legend(loc="best")
                axs[1].grid(True, linestyle="--", linewidth=0.6)

                plt.tight_layout()
    plt.show()


# ----------------------------------------------------
# Entry point
# ----------------------------------------------------
if __name__ == "__main__":
    main()
