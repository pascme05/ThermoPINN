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

import os
import torch.optim as optim
import matplotlib.pyplot as plt
from dev.estGC import *
from src.model import *


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

    # Fixed Thermal parameters
    Rth = []                                                                                                             # Thermal resistance foster network [K/W]
    Cth = []                                                                                                             # Thermal capacitance foster network [Ws/K]

    # Motor parameters
    Rs = 14.1e-3                                                                                                         # Stator resistance [Ohm]
    alpha = 0.00393                                                                                                      # Temperature coefficient of resistance [1/°C]
    Tref = 20                                                                                                            # Reference temperature for Rs [°C]
    n_max = 6000                                                                                                         # Maximum motor speed [rpm]
    beta_1 = 0.315                                                                                                       # Parameter for frequency losses (linear)
    beta_2 = 0.616                                                                                                       # Parameter for frequency losses (quadratic)
    k1 = 0.719
    k2 = -0.059
    k3 = 0.069
    k4 = -0.619
    alpha_Fe = -0.71

    # Training hyperparameters
    seq_len = 600                                                                                                        # Sequence length (timesteps per training sample)
    stride = 50                                                                                                          # Step size between training sequences
    batch_size = 32                                                                                                      # Batch size for training
    hidden_dim = 8                                                                                                       # Hidden units in LSTM layers
    num_layers = 2                                                                                                       # Number of stacked LSTM layers
    lr = 1e-3                                                                                                            # Learning rate for optimizer
    epochs = 20                                                                                                          # Maximum number of training epochs
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
    dt_s = 1.0

    for id_sel in id_list:
        df_step = df[df["id"] == id_sel].copy().head(3500)
        time_step = df_step["time"].values - df_step["time"].values[0]
        dt_s = np.mean(np.diff(time_step))

        # Temperatures (replace with your stator and rotor columns)
        T_s = df_step["Tsw"].values
        T_r = df_step["Trm"].values
        T_amb = df_step["Tc"].values

        Is = df_step["Is"].values / np.sqrt(2)
        Id = df_step["Id"].values
        Iq = df_step["Iq"].values
        Us = df_step["Us"].values / np.sqrt(2)
        Ud = df_step["Ud"].values
        Uq = df_step["Uq"].values
        Wm = df_step["Wm"].values
        Mm = df_step["Mm"].values

        f1 = (1 + alpha * (T_s - Tref))
        f2 = 1 + beta_1 * (Wm / n_max) + beta_2 * (Wm / n_max) ** 2
        P_s = 3 * Rs * Is ** 2 * f1 * f2
        P_el = -3/2 * (Ud*Id + Uq*Iq)
        P_mech = 2 * np.pi * Wm / 60 * Mm
        P_res = (P_el - P_mech) - P_s
        k = k1 + k2 * Is + k3 * Wm + k4 * Is * Wm
        P_r = (1 - k) * P_res * (1 + alpha_Fe * (T_r - Tref))

        T_nodes = np.stack([T_s, T_r], axis=1)
        P_inputs = np.stack([P_s, P_r], axis=1)

        C_diag, G, _, _ = est_CG(T_nodes, P_inputs, T_amb, dt_s)

        id_data.append({
            "id": id_sel,
            "Is": np.mean(Is),
            "Wm": np.mean(Wm),
            "G_ss": G[0, 0],
            "G_sr": G[0, 1],
            "G_rs": G[1, 0],
            "G_rr": G[1, 1],
            "C_s": C_diag[0],
            "C_r": C_diag[1],
        })

    df_ident = pd.DataFrame(id_data)
    print(df_ident.head())

    C_hat = np.array([df_ident["C_s"].mean(), df_ident["C_r"].mean()])
    G_hat = np.array([[df_ident["G_ss"].mean(), df_ident["G_sr"].mean()],
                      [df_ident["G_rs"].mean(), df_ident["G_rr"].mean()]])

    # -------------------------------
    # Feature preparation
    # -------------------------------
    # Losses
    P_el = 3 * df["Is"] * df["Us"]
    P_mech = 2 * np.pi * df["Wm"] / 60 * df["Mm"]
    Pv_sw = 3 * Rs * df["Is"] ** 2
    P_res = (P_el - P_mech) - Pv_sw
    k = k1 + k2 * df["Is"] + k3 * df["Wm"] + k4 * df["Is"] * df["Wm"]
    Pv_rm = (1 - k) * P_res
    P_in = np.stack([Pv_sw, Pv_rm], axis=1)

    # Temperatures
    T_pred = calc_T(C_hat, G_hat, P_in, df["Tc"], T0=None, dt=1.0)

    # Calc features
    f1 = (1 + alpha * (df["Tc"] - Tref))
    f2 = 1 + beta_1 * (df["Wm"] / n_max) + beta_2 * (df["Wm"] / n_max) ** 2
    f3 = (1 + alpha_Fe * (df["Tc"] - Tref))
    df["Pv_s"] = Pv_sw * f1 * f2
    df["Pv_r"] = Pv_rm * f3
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
    X_train, T_train, Tamb_train = normalize(df_train, feature_cols, X_mean, X_std, T_max, T_min)
    X_val, T_val, Tamb_val = normalize(df_val, feature_cols, X_mean, X_std, T_max, T_min)

    # Calc Power
    P_train = np.zeros((len(train_ids), 2))
    P_val = np.zeros((len(val_ids), 2))
    P_train[:, 0] = df_train["Pv_s"].values()
    P_train[:, 1] = df_train["Pv_r"].values()
    P_val[:, 0] = df_val["Pv_s"].values()
    P_val[:, 1] = df_val["Pv_r"].values()

    # -------------------------------
    # DataLoaders
    # -------------------------------
    train_loader = prepare_loader(X_train, T_train, P_train, Tamb_train, df_train["time_id"].values,
                                  df_train["T0"].values, seq_len, stride, batch_size, DEVICE,
                                  df[df["id"].isin(test_ids)]["id"].to_numpy(), shuffle=False)
    val_loader = prepare_loader(X_val, T_val, P_val, Tamb_val, df_val["time_id"].values,
                                df_val["T0"].values, seq_len, stride, batch_size, DEVICE,
                                df[df["id"].isin(test_ids)]["id"].to_numpy(), shuffle=False)

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
                                                           G_hat, C_hat, Tambb, T_min, T_max,
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
                                                                    G_hat, C_hat, Tambv, T_min, T_max,
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
                torch.save(model.state_dict(), "../mdl/mdl_opti_pinn.pt")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    # -------------------------------
    # Test evaluation
    # -------------------------------
    model.load_state_dict(torch.load("../mdl/mdl_opti_pinn.pt"))
    model.eval()

    for sid in test_ids:
        print(f"\n🔹 Evaluating Test Session ID: {sid}")

        # Select session-specific data
        df_session = df_test[df_test["id"] == sid].copy()
        if df_session.empty:
            print(f"⚠️ No data found for session {sid}, skipping.")
            continue

        # Prepare physics-informed scaling
        P_test = np.stack([df_session["Pv_s"], df_session["Pv_r"]], axis=1)

        # RC model prediction
        T_pred_rc = calc_T(C_hat, G_hat, P_in, df["Tc"], T0=None, dt=1.0)

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
