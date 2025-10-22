#######################################################################################################################
#######################################################################################################################
# Title:        ThermoPINN
# Topic:        Physics Informed Neural Network (PINN) for thermal modeling
# File:         mainCool
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
from src.model import *


#######################################################################################################################
# Main workflow
#######################################################################################################################
def main():
    # ==============================================================================
    # Parameters
    # ==============================================================================
    # -------------------------------
    # Seeds
    # -------------------------------
    np.random.seed(42)

    # -------------------------------
    # Config / Parameters
    # -------------------------------
    # Settings
    TRAIN_MODEL = True
    ENABLE_PLOTS = True
    TRANSFER = False
    SPLIT_ID = True
    RESAMPLE = False

    # Path
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MDL_NAME = "mdl/mdl_cool.pt"
    DATA_PATH = os.path.join(BASE_DIR, "data")

    # -------------------------------
    # Data
    # -------------------------------
    # Data files
    DATA_TRAIN = ["data1"]
    DATA_VAL = "data2"
    DATA_TEST = "data3"

    # Data Split
    test_split = 0.2
    val_split = 0.1

    # Data IDs Non-Transfer
    test_ids = [60, 62, 74]  # IDs used for test set evaluation
    val_ids = [10, 48, 63]  # IDs used for validation set selection

    # Select features
    selX = ["Ta", "I_HV", "U_HV", "v", "Vol"]
    selY = "Tc"
    selP = "Pv"
    selt = "time_id"
    selT = "Ta"
    selT0 = "Tc0"

    # -------------------------------
    # Fixed Thermal parameters
    # -------------------------------
    Rth = 1/1200                                                                                                         # Thermal resistance foster network [K/W]
    Cth = 20 * 4200                                                                                                      # Thermal capacitance foster network [Ws/K]

    # -------------------------------
    # General Parameters
    # -------------------------------
    sampleFactor = 10                                                                                                    # integer factor for downsampling
    sample_cols = ["time", "time_id"]                                                                                    # data columns that should be downsampled (all others are averaged)

    # -------------------------------
    # Training hyperparameters
    # -------------------------------
    seq_len = 1300                                                                                                       # Sequence length (timesteps per training sample)
    stride = 50                                                                                                          # Step size between training sequences
    batch_size = 32                                                                                                      # Batch size for training
    hidden_dim = 256                                                                                                     # Hidden units in LSTM layers
    num_layers = 2                                                                                                       # Number of stacked LSTM layers
    lr = 1e-3                                                                                                            # Learning rate for optimizer
    epochs = 100                                                                                                         # Maximum number of training epochs
    lambda_phys = 0.1                                                                                                    # Weight for physics-informed loss term
    lambda_init = 0.0                                                                                                    # Weight for initial condition loss (currently unused)
    patience = 10                                                                                                        # Early stopping patience (epochs without improvement)

    # ==============================================================================
    # Load data
    # ==============================================================================
    # -------------------------------
    # Transfer
    # -------------------------------
    if TRANSFER:
        # Train
        tempPath = os.path.join(DATA_PATH, DATA_TRAIN[0])
        df_train = pd.read_csv(tempPath)
        for i in range(1, len(DATA_TRAIN)):
            tempPath = os.path.join(DATA_PATH, DATA_TRAIN[i])
            df_train.append(pd.read_csv(tempPath))

        # Test
        tempPath = os.path.join(DATA_PATH, DATA_TEST)
        df_test = pd.read_csv(tempPath)

        # Validation
        tempPath = os.path.join(DATA_PATH, DATA_VAL)
        df_val = pd.read_csv(tempPath)

    # -------------------------------
    # Non-Transfer
    # -------------------------------
    else:
        # Load data train
        tempPath = os.path.join(DATA_PATH, DATA_TRAIN[0])
        df = pd.read_csv(tempPath)

        # IDs
        if SPLIT_ID:
            train_ids = [i for i in df["id"].unique() if i not in test_ids + val_ids]
        else:
            unique_ids = df["id"].unique()
            selected_ids  = np.random.choice(unique_ids, size=int(len(unique_ids) * (test_split + val_split)), replace=False)
            n_train = int(len(selected_ids) * test_split)
            test_ids = selected_ids[:n_train]
            val_ids = selected_ids[n_train:]
            train_ids = [i for i in df["id"].unique() if i not in test_ids + val_ids]

        # Split data
        df_train = df[df["id"].isin(train_ids)].copy()
        df_val = df[df["id"].isin(val_ids)].copy()
        df_test = df[df["id"].isin(test_ids)].copy()

    # -------------------------------
    # Sampling
    # -------------------------------
    Ts = df_train["time"].values[1] - df_train["time"].values[0]

    # ==============================================================================
    # Pre-Processing
    # ==============================================================================
    # -------------------------------
    # Resample
    # -------------------------------
    if RESAMPLE:
        # Groups
        Ts = Ts * sampleFactor
        mean_cols = [c for c in df_train.columns if c not in sample_cols]

        # Sample
        df_train = (
            df_train.groupby(df_train.index // sampleFactor)
            .agg({**{col: 'mean' for col in mean_cols},
                  **{col: 'last' for col in sample_cols}})
            .reset_index(drop=True))

        df_test = (
            df_test.groupby(df_test.index // sampleFactor)
            .agg({**{col: 'mean' for col in mean_cols},
                  **{col: 'last' for col in sample_cols}})
            .reset_index(drop=True))

        df_val = (
            df_val.groupby(df_val.index // sampleFactor)
            .agg({**{col: 'mean' for col in mean_cols},
                  **{col: 'last' for col in sample_cols}})
            .reset_index(drop=True))

    # -------------------------------
    # Feature Preparation
    # -------------------------------
    # Mechanical Power
    df_train["P_mech"] = 2 * np.pi * df_train["M_f"] * df_train["n_f"] / 60 + 2 * np.pi * df_train["M_r"] * df_train["n_r"]
    df_test["P_mech"] = 2 * np.pi * df_test["M_f"] * df_test["n_f"] / 60 + 2 * np.pi * df_test["M_r"] * df_test["n_r"]
    df_val["P_mech"] = 2 * np.pi * df_val["M_f"] * df_val["n_f"] / 60 + 2 * np.pi * df_val["M_r"] * df_val["n_r"]

    # Electrical Power
    df_train["P_el"] = df_train["I_HV"] * df_train["V_LV"]
    df_test["P_el"] = df_test["I_HV"] * df_test["V_LV"]
    df_val["P_el"] = df_val["I_HV"] * df_val["V_LV"]

    # Losses
    df_train["Pv"] = (df_train["P_mech"] - df_train["P_el"]).abs()
    df_test["Pv"] = (df_test["P_mech"] - df_test["P_el"]).abs()
    df_val["Pv"] = (df_val["P_mech"] - df_val["P_el"]).abs()

    # -------------------------------
    # Split data
    # -------------------------------
    X_train, X_test, X_val = df_train[selX], df_test[selX], df_val[selX]
    y_train, y_test, y_val = df_train[selY], df_test[selY], df_val[selY]
    P_train, P_test, P_val = df_train[selP], df_test[selP], df_val[selP]
    T_train, T_test, T_val = df_train[selT], df_test[selT], df_val[selT]
    T0_train, T0_test, T0_val = df_train[selT0], df_test[selT0], df_val[selT0]
    t_train, t_test, t_val = df_train[selt], df_test[selt], df_val[selt]
    id_train, id_test, id_val = (df_train["id"], df_test["id"], df_val["id"])

    # -------------------------------
    # Normalize and Scale
    # -------------------------------
    # Norm values
    X_mean, X_std = X_train.mean(), X_train.std() + 1e-8
    y_min, y_max = y_train.min(), y_train.max()

    # Normalize
    X_train = (X_train - X_mean) / X_std
    y_train = (y_train - y_min) / (y_max - y_min)

    # ==============================================================================
    # Training
    # ==============================================================================
    # -------------------------------
    # DataLoaders
    # -------------------------------
    train_loader = prepare_loader(X_train, y_train, P_train, T_train, t_train, T0_train,
                                  seq_len, stride, batch_size, DEVICE, id_train, shuffle=False)
    val_loader = prepare_loader(X_val, y_val, P_val, T_val, t_val, T0_val,
                                seq_len, stride, batch_size, DEVICE, id_val, shuffle=False)

    # -------------------------------
    # Model setup
    # -------------------------------
    model = LSTM_PINN(input_dim=len(selX), hidden_dim=hidden_dim, num_layers=num_layers).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
    dt_torch = torch.tensor(Ts, dtype=torch.float32, device=DEVICE)

    # -------------------------------
    # Init
    # -------------------------------
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
                                                           Rth, Cth, Tambb, y_min, y_max,
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
                                                                    Rth, Cth, Tambv, y_min, y_max,
                                                                    lambda_phys, lambda_init)
                    val_total += v_loss.item()
                    val_data += v_dmse
                    val_phys += v_pmse
                    val_init += v_imse

            train_loss = total_loss / len(train_loader)
            val_loss = val_total / len(val_loader)
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']

            print(f"Epoch {ep + 1:03d} | LR={current_lr:.6f} | "
                  f"Train={train_loss:.6f} | Val={val_loss:.6f} | "
                  f"Data={val_data / len(val_loader):.6f} | Phys={val_phys / len(val_loader):.6f} | "
                  f"Init={val_init / len(val_loader):.6f}")

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

    # -------------------------------
    # Test evaluation
    # -------------------------------
    model.load_state_dict(torch.load(MDL_NAME))
    model.eval()

    for sid in test_ids:
        print(f"\n🔹 Evaluating Test Session ID: {sid}")

        # Select session-specific data
        df_session = df_test[df_test["id"] == sid].copy()
        if df_session.empty:
            print(f"⚠️ No data found for session {sid}, skipping.")
            continue

        # Select data
        X_test_temp = X_test[X_test["id"] == sid].values
        T_test_temp = T_test[T_test["id"] == sid].values

        with torch.no_grad():
            X_tensor = torch.tensor(X_test_temp, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            T_pred = model(X_tensor).cpu().numpy().reshape(-1)

            # Inverse normalization
            T_true = T_test_temp * (y_max - y_min) + y_min
            T_pred = T_pred * (y_max - y_min) + y_min

            # Compute errors
            err = T_pred - T_true

            # Error Metrics
            mseErr = np.mean(err ** 2)
            maeErr = np.mean(np.abs(err))
            maxErr = np.max(np.abs(err))

            print(f"NN MSE (°C): {mseErr:.2f}")
            print(f"NN MAE (°C): {maeErr:.2f}")
            print(f"NN MAX (°C): {maxErr:.2f}")

            # -------------------------------
            # Plotting per session
            # -------------------------------
            if ENABLE_PLOTS:
                time = np.linspace(0, (len(err)*Ts - 1) / 60, len(err))  # time in minutes
                fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
                fig.suptitle(f"Session ID {sid} – Model Comparison", fontsize=12, fontweight="bold")

                # ▪️ Top: Current Voltage
                axs[0].plot(time, df_session["Is"].values, label="Current", linewidth=2)
                axs[0].plot(time, df_session["Us"].values, label="Voltage", linewidth=2)
                axs[0].set_ylabel("Current [A] / Voltage [V]")
                axs[0].legend(loc="best")
                axs[0].grid(True, linestyle="--", linewidth=0.6)

                # ▪️ Mid: Temperatures
                axs[1].plot(time, T_true, label="Measured", linewidth=2)
                axs[1].plot(time, T_pred, label="Pred RC", linewidth=2)
                axs[1].plot(time, df_session["Tc"].values, label="Coolant", linewidth=2)
                axs[1].set_ylabel("Temperature [°C]")
                axs[1].legend(loc="best")
                axs[1].grid(True, linestyle="--", linewidth=0.6)

                # ▪️ Bottom: Errors
                axs[2].plot(time, err, label="Error")
                axs[2].axhline(0, color="black", linewidth=1)
                axs[2].set_xlabel("Time [min]")
                axs[2].set_ylabel("Error [°C]")
                axs[2].legend(loc="best")
                axs[2].grid(True, linestyle="--", linewidth=0.6)

                plt.tight_layout()
    plt.show()


# ----------------------------------------------------
# Entry point
# ----------------------------------------------------
if __name__ == "__main__":
    main()
