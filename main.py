#######################################################################################################################
#######################################################################################################################
# Title:        ThermoPINN
# Topic:        Physics Informed Neural Network (PINN) for thermal modeling
# File:         main
# Date:         17.10.2025
# Author:       Dr. Pascal A. Schirmer
# Version:      V.0.1
# Copyright:    Pascal Schirmer
#######################################################################################################################
#######################################################################################################################


#######################################################################################################################
# Import external libs
#######################################################################################################################
import os
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from src.model import *


#######################################################################################################################
# Main workflow
#######################################################################################################################
def main():
    # ==============================================================================
    # Config / Parameters
    # ==============================================================================
    # -------------------------------
    # Config / Parameters
    # -------------------------------
    TRAIN_MODEL = True
    ENABLE_PLOTS = True

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "motor_temp.csv")
    MDL_NAME = "mdl/mdl_best_pinn.pt"

    # -------------------------------
    # Fixed Thermal parameters
    # -------------------------------
    Rth = []                                                                                                             # Thermal resistance foster network [K/W]
    Cth = []                                                                                                             # Thermal capacitance foster network [Ws/K]

    # -------------------------------
    # Motor parameters
    # -------------------------------
    Rs = 14.1e-3                                                                                                         # Stator resistance [Ohm]
    alpha = 0.00393                                                                                                      # Temperature coefficient of resistance [1/°C]
    Tref = 20                                                                                                            # Reference temperature for Rs [°C]
    n_max = 6000                                                                                                         # Maximum motor speed [rpm]
    beta_1 = 0.315                                                                                                       # Parameter for frequency losses (linear)
    beta_2 = 0.616                                                                                                       # Parameter for frequency losses (quadratic)

    # -------------------------------
    # Training hyperparameters
    # -------------------------------
    seq_len = 1300                                                                                                        # Sequence length (timesteps per training sample)
    stride = 50                                                                                                          # Step size between training sequences
    batch_size = 32                                                                                                      # Batch size for training
    hidden_dim = 256                                                                                                       # Hidden units in LSTM layers
    num_layers = 2                                                                                                       # Number of stacked LSTM layers
    lr = 1.67e-3                                                                                                            # Learning rate for optimizer
    epochs = 100                                                                                                          # Maximum number of training epochs
    lambda_phys = 0.02                                                                                                    # Weight for physics-informed loss term
    lambda_init = 0.50                                                                                                   # Weight for initial condition loss (currently unused)
    patience = 10                                                                                                        # Early stopping patience (epochs without improvement)

    # ==============================================================================
    # Data
    # ==============================================================================
    # -------------------------------
    # Dataset split IDs
    # -------------------------------
    test_ids = [60, 62, 74]                                                                                              # IDs used for test set evaluation
    val_ids = [10, 48, 63]                                                                                               # IDs used for validation set selection

    # -------------------------------
    # Load CSV data
    # -------------------------------
    df = pd.read_csv(DATA_PATH)

    # ==============================================================================
    # RC Identification
    # ==============================================================================
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

    # Polynomial fit (optional, can be used later)
    poly_R = make_pipeline(PolynomialFeatures(2), LinearRegression())
    poly_R.fit(df_ident[["Is", "Wm"]], df_ident["R"])
    poly_C = make_pipeline(PolynomialFeatures(2), LinearRegression())
    poly_C.fit(df_ident[["Is", "Wm"]], df_ident["C"])

    if not Rth:
        R_hat = df_ident["R"].mean()
        C_hat = df_ident["C"].mean()
        print(f"Identified Average: R_th = {R_hat:.4f} K/W, C_th = {C_hat:.2f} J/K")
    else:
        R_hat = Rth[0]
        C_hat = Cth[0]
        print(f"Defined Average: R_th = {R_hat:.4f} K/W, C_th = {C_hat:.2f} J/K")

    # ==============================================================================
    # Pre-processing
    # ==============================================================================
    # -------------------------------
    # Feature preparation
    # -------------------------------
    f1 = (1 + alpha * (df["Tc"] - Tref))
    f2 = 1 + beta_1 * (df["Wm"] / n_max) + beta_2 * (df["Wm"] / n_max) ** 2
    df["Pv_s"] = 3 * Rs * (df["Is"] / np.sqrt(2)) ** 2 * f1 * f2
    df["Sel"] = 3/2 * df["Is"] * df["Us"]
    df["SelI"] = df["Sel"] * df["Is"]
    df["SelW"] = df["Sel"] * df["Wm"]

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
    # Features
    feature_cols = [c for c in df.columns if c not in ["id", "time", "time_id", "T0", "Tsw", "Tst", "Tso", "Trm"]]

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
    P_train *= (1 + alpha * (df_train["Tsw"] - Tref)) * (1 + beta_1 * (df_train["Wm"] / n_max) + beta_2 * (df_train["Wm"] / n_max) ** 2)
    P_val *= (1 + alpha * (df_val["Tsw"] - Tref)) * (1 + beta_1 * (df_val["Wm"] / n_max) + beta_2 * (df_val["Wm"] / n_max) ** 2)

    # -------------------------------
    # DataLoaders
    # -------------------------------
    train_loader = prepare_loader(X_train, T_train, P_train, Tamb_train, df_train["time_id"].values,
                                  df_train["T0"].values, seq_len, stride, batch_size, DEVICE,
                                  df[df["id"].isin(train_ids)]["id"].to_numpy(), shuffle=False)
    val_loader = prepare_loader(X_val, T_val, P_val, Tamb_val, df_val["time_id"].values,
                                df_val["T0"].values, seq_len, stride, batch_size, DEVICE,
                                df[df["id"].isin(test_ids)]["id"].to_numpy(), shuffle=False)

    # ==============================================================================
    # Training
    # ==============================================================================
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
                torch.save(model.state_dict(), MDL_NAME)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

    # ==============================================================================
    # Testing
    # ==============================================================================
    # -------------------------------
    # Setup
    # -------------------------------
    model.load_state_dict(torch.load(MDL_NAME))
    model.eval()
    T_pred_all = np.empty((0,), float)
    T_true_all = np.empty((0,), float)

    # -------------------------------
    # Loop
    # -------------------------------
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
        X_test, T_test, _ = normalize(df_session, feature_cols, X_mean, X_std, T_max, T_min)

        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            T_pred_nn = model(X_tensor).cpu().numpy().reshape(-1)

            # Inverse normalization
            T_true_nn = T_test * (T_max - T_min) + T_min
            T_pred_nn_phys = T_pred_nn * (T_max - T_min) + T_min

            # Append Total
            T_pred_all = np.append(T_pred_all, T_pred_nn_phys, axis=0)
            T_true_all = np.append(T_true_all, T_true_nn, axis=0)

            # Compute errors
            err_rc = T_pred_rc - T_true_nn
            err_nn = T_pred_nn_phys - T_true_nn

            # Error Metrics
            mse_test_rc = np.mean(err_rc ** 2)
            mse_test_nn = np.mean(err_nn ** 2)
            mae_test_rc = np.mean(np.abs(err_rc))
            mae_test_nn = np.mean(np.abs(err_nn))
            max_test_rc = np.max(np.abs(err_rc))
            max_test_nn = np.max(np.abs(err_nn))

            print(f"RC MSE (°C): {mse_test_rc:.2f}")
            print(f"RC MAE (°C): {mae_test_rc:.2f}")
            print(f"RC MAX (°C): {max_test_rc:.2f}")
            print(f"NN MSE (°C): {mse_test_nn:.2f}")
            print(f"NN MAE (°C): {mae_test_nn:.2f}")
            print(f"NN MAX (°C): {max_test_nn:.2f}")

            # Plotting per session
            if ENABLE_PLOTS:
                time = np.linspace(0, (len(err_rc) - 1) / 60, len(err_rc))  # time in minutes
                fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
                fig.suptitle(f"Session ID {sid} – Model Comparison", fontsize=12, fontweight="bold")

                # ▪️ Top: Current Voltage
                axs[0].plot(time, df_session["Is"].values, label="Current", linewidth=2)
                axs[0].plot(time, df_session["Us"].values, label="Voltage", linewidth=2)
                axs[0].set_ylabel("Current [A] / Voltage [V]")
                axs[0].legend(loc="best")
                axs[0].grid(True, linestyle="--", linewidth=0.6)

                # ▪️ Mid: Temperatures
                axs[1].plot(time, T_true_nn, label="Measured", color="black", linewidth=2)
                axs[1].plot(time, T_pred_rc, label="Pred RC", linewidth=1.8)
                axs[1].plot(time, T_pred_nn_phys, label="Pred NN", linewidth=1.8)
                axs[1].plot(time, df_session["Tc"].values, label="Coolant", linewidth=1.8)
                axs[1].set_ylabel("Temperature [°C]")
                axs[1].legend(loc="best")
                axs[1].grid(True, linestyle="--", linewidth=0.6)

                # ▪️ Bottom: Errors
                axs[2].plot(time, err_rc, label="RC Error")
                axs[2].plot(time, err_nn, label="NN Error")
                axs[2].axhline(0, color="black", linewidth=1)
                axs[2].set_xlabel("Time [min]")
                axs[2].set_ylabel("Error [°C]")
                axs[2].legend(loc="best")
                axs[2].grid(True, linestyle="--", linewidth=0.6)

                plt.tight_layout()

    # Total Error
    err_nn_all = T_pred_all - T_true_all

    # Error Metrics
    mse_test_nn_all = np.mean(err_nn_all ** 2)
    mae_test_nn_all = np.mean(np.abs(err_nn_all))
    max_test_nn_all = np.max(np.abs(err_nn_all))

    print(f"\n🔹 Evaluating Test Total")
    print(f"NN MSE (°C): {mse_test_nn_all:.2f}")
    print(f"NN MAE (°C): {mae_test_nn_all:.2f}")
    print(f"NN MAX (°C): {max_test_nn_all:.2f}")

    # Show plot
    plt.show()


#######################################################################################################################
# Entry point
#######################################################################################################################
if __name__ == "__main__":
    main()
