# ----------------------------------------------------
# pinn_thermal_lstm_optuna.py
# Author: Pascal Schirmer / ChatGPT-5
# Date: 2025-10-14
# Description:
#   Physics-Informed Neural Network (PINN) using LSTM for
#   thermal modeling with automated hyperparameter optimization
#   via Optuna.
# ----------------------------------------------------

import os
import optuna
import torch.optim as optim
from src.model import *

# ----------------------------------------------------
# Global setup
# ----------------------------------------------------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = "data/motor_temp.csv"


# ----------------------------------------------------
# Objective function for Optuna
# ----------------------------------------------------
def objective(trial):
    # -------------------------------
    # Hyperparameter suggestions
    # -------------------------------
    hidden_dim = trial.suggest_categorical("hidden_dim", [8, 16, 32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 3)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    seq_len = trial.suggest_int("seq_len", 200, 800, step=100)
    lambda_phys = trial.suggest_float("lambda_phys", 0.01, 1.0, log=True)
    lambda_init = trial.suggest_float("lambda_init", 0.0, 0.5)
    epochs = 10
    stride = 10

    # -------------------------------
    # Fixed motor/thermal parameters
    # -------------------------------
    Rs = 14.1e-3
    alpha = 0.00393
    Tref = 20
    n_max = 6000
    beta_1 = 0.315
    beta_2 = 0.616
    dt_s = 1.0

    # -------------------------------
    # Load data once globally (cache)
    # -------------------------------
    global df_cache
    if "df_cache" not in globals():
        df_cache = pd.read_csv(DATA_PATH)

    df = df_cache.copy()

    # -------------------------------
    # RC identification
    # -------------------------------
    id_list = [2, 3, 5, 7, 8, 9, 12, 13, 14, 15, 16, 17, 19, 21, 24]
    id_data = []
    for id_sel in id_list:
        df_step = df[df["id"] == id_sel].copy().head(3500)
        time_step = df_step["time"].values - df_step["time"].values[0]
        T_amb = df_step["Tc"].values
        T_step = df_step["Tsw"].values
        Is = df_step["Is"].values / np.sqrt(2)
        Wm = df_step["Wm"].values

        f1 = (1 + alpha * (T_step - Tref))
        f2 = 1 + beta_1 * (Wm / n_max) + beta_2 * (Wm / n_max) ** 2
        P_step = 3 * Rs * Is ** 2 * f1 * f2
        dt_s = np.mean(np.diff(time_step))

        R_fit, C_fit = identify_rc(P_step.flatten(), T_step, T_amb, dt_s)
        id_data.append({"id": id_sel, "Is": np.mean(Is), "Wm": np.mean(Wm),
                        "Pv": np.max(P_step), "R": R_fit, "C": C_fit})

    df_ident = pd.DataFrame(id_data)
    R_hat, C_hat = df_ident["R"].mean(), df_ident["C"].mean()

    # -------------------------------
    # Feature computation
    # -------------------------------
    # Calc features
    f1 = (1 + alpha * (df["Tc"] - Tref))
    f2 = 1 + beta_1 * (df["Wm"] / n_max) + beta_2 * (df["Wm"] / n_max) ** 2
    df["Pv_s"] = 3 * Rs * (df["Is"] / np.sqrt(2)) ** 2 * f1 * f2
    df["Sel"] = 3 / 2 * df["Is"] * df["Us"]
    df["SelI"] = df["Sel"] * df["Is"]
    df["SelW"] = df["Sel"] * df["Wm"]

    # Remove features
    feature_cols = [c for c in df.columns if c not in ["id", "time", "time_id", "T0", "Tsw", "Tst", "Tso", "Trm"]]

    # -------------------------------
    # Data split
    # -------------------------------
    test_ids = [60, 62, 74]
    val_ids = [10, 48, 63]
    train_ids = [i for i in df["id"].unique() if i not in test_ids + val_ids]
    df_train = df[df["id"].isin(train_ids)].copy()
    df_val = df[df["id"].isin(val_ids)].copy()

    # -------------------------------
    # Normalization
    # -------------------------------
    # Norm values
    X_mean, X_std = df_train[feature_cols].mean(), df_train[feature_cols].std() + 1e-8
    T_min, T_max = df_train["Tsw"].min(), df_train["Tsw"].max()

    # Normalize
    X_train, T_train, Tamb_train = normalize(df_train, feature_cols, X_mean, X_std, T_max, T_min)
    X_val, T_val, Tamb_val = normalize(df_val, feature_cols, X_mean, X_std, T_max, T_min)

    # Calc Power
    P_train = 3 * Rs * (df_train["Is"] / np.sqrt(2)) ** 2
    P_val = 3 * Rs * (df_val["Is"] / np.sqrt(2)) ** 2

    # Scale Temperature-dependent Power
    P_train *= (1 + alpha * (df_train["Tsw"] - Tref)) * (
                1 + beta_1 * (df_train["Wm"] / n_max) + beta_2 * (df_train["Wm"] / n_max) ** 2)
    P_val *= (1 + alpha * (df_val["Tsw"] - Tref)) * (
                1 + beta_1 * (df_val["Wm"] / n_max) + beta_2 * (df_val["Wm"] / n_max) ** 2)

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
    patience = 5
    patience_counter = 0

    # -------------------------------
    # Training loop
    # -------------------------------
    for ep in range(epochs):
        model.train()
        total_loss = 0

        for Xb, Tb, Pb, Tambb, t_seq, T0b in train_loader:
            optimizer.zero_grad()
            loss, _, _, _ = pinn_loss_lstm(model, Xb, Tb, Pb, t_seq, T0b, dt_torch,
                                           R_hat, C_hat, Tambb, T_min, T_max,
                                           lambda_phys, lambda_init)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_total = 0
        with torch.no_grad():
            for Xv, Tv, Pv, Tambv, tv_seq, T0v in val_loader:
                v_loss, _, _, _ = pinn_loss_lstm(model, Xv, Tv, Pv, tv_seq, T0v, dt_torch,
                                                 R_hat, C_hat, Tambv, T_min, T_max,
                                                 lambda_phys, lambda_init)
                val_total += v_loss.item()

        val_loss = val_total / len(val_loader)
        scheduler.step(val_loss)

        # Optuna pruning
        trial.report(val_loss, ep)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return best_val_loss

