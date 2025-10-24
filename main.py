#######################################################################################################################
# Title:        ThermoPINN
# Topic:        Physics Informed Neural Network (PINN) for thermal modeling
# File:         main
# Date:         22.10.2025
# Author:       Dr. Pascal A. Schirmer
# Version:      V.0.2 (with full k-fold CV and RC identification)
#######################################################################################################################
import os
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import make_pipeline
from src.helpTrain import *


#######################################################################################################################
# Main workflow
#######################################################################################################################
def main():
    ###################################################################################################################
    # Configuration
    ###################################################################################################################
    # ==============================================================================
    # Settings
    # ==============================================================================
    TRAIN_MODEL = True
    ENABLE_PLOTS = True
    N_FOLDS = 1

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "measures_v2.csv")
    MDL_NAME = os.path.join(BASE_DIR, "mdl", "mdl_best_pinn.pt")

    # ==============================================================================
    # General Parameter
    # ==============================================================================
    W = 5                                                                                                               # Window length for filtering data

    # ==============================================================================
    # Physical parameters
    # ==============================================================================
    Rs = 14.1e-3                                                                                                         # Stator resistance [Ohm]
    alpha = 0.00393                                                                                                      # Temperature coefficient of resistance [1/°C]
    Tref = 20                                                                                                            # Reference temperature for Rs [°C]
    n_max = 6000                                                                                                         # Maximum motor speed [rpm]
    beta_1 = 0.315                                                                                                       # Parameter for frequency losses (linear)
    beta_2 = 0.616                                                                                                       # Parameter for frequency losses (quadratic)

    # ==============================================================================
    # Training hyperparameters
    # ==============================================================================
    seq_len = 1200                                                                                                       # Sequence length (timesteps per training sample)
    stride = 100                                                                                                          # Step size between training sequences
    batch_size = 32                                                                                                      # Batch size for training
    hidden_dim = 256                                                                                                     # Hidden units in LSTM layers
    num_layers = 2                                                                                                       # Number of stacked LSTM layers
    dropout = 0.3                                                                                                        # Dropout for the LSTM layer
    lr = 1.67e-3                                                                                                         # Learning rate for optimizer
    epochs = 100                                                                                                         # Maximum number of training epochs
    lambda_phys = 0.06                                                                                                   # Weight for physics-informed loss term
    lambda_init = 0.50                                                                                                   # Weight for initial condition loss (currently unused)
    patience = 10                                                                                                        # Early stopping patience (epochs without improvement)

    # ==============================================================================
    # Dataset IDs
    # ==============================================================================
    test_ids = [60, 62, 74]
    val_ids = [10, 48, 63]
    id_list = [2, 3, 5, 7, 8, 9, 12, 13, 14, 15, 16, 17, 19, 21, 24]

    # ==============================================================================
    # Input and Output Select
    # ==============================================================================
    selX = ["Ta", "Tc", "Is", "Id", "Iq", "Us", "Ud", "Uq", "Wm", "Mm", "Sel", "SelI", "SelW"]
    selY = "Tsw"
    selR = "Tc"

    ###################################################################################################################
    # Load data
    ###################################################################################################################
    # ==============================================================================
    # Input and Output Select
    # ==============================================================================
    df = pd.read_csv(DATA_PATH)

    # ==============================================================================
    # Rename data
    # ==============================================================================
    df = df.rename(columns={'ambient': 'Ta', 'coolant': 'Tc', 'i_d': 'Id', 'i_q': 'Iq', 'u_d': 'Ud', 'u_q': 'Uq',
                            'profile_id': 'id', 'torque': 'Mm', 'motor_speed': 'Wm', 'stator_winding': 'Tsw'})

    # ==============================================================================
    # Calculate features
    # ==============================================================================
    df["Is"] = (df["Id"] ** 2 + df["Iq"] ** 2) ** 0.5
    df["Us"] = (df["Ud"] ** 2 + df["Uq"] ** 2) ** 0.5
    df["Sel"] = 3 / 2 * df["Is"] * df["Us"]
    df["SelI"] = df["Sel"] * df["Is"]
    df["SelW"] = df["Sel"] * df["Wm"]
    df['T0'] = df.groupby('id')['Tsw'].transform('first')
    df["time"] = np.linspace(0, (len(df["Is"])*0.5 - 1/2), len(df["Is"]))
    df['time_id'] = df.groupby('id')['time'].transform(lambda x: x - x.iloc[0])

    # ==============================================================================
    # Time vector
    # ==============================================================================
    dt_s = df["time"].values[1] - df["time"].values[0]

    ###################################################################################################################
    # RC Identification
    ###################################################################################################################
    # ==============================================================================
    # Init
    # ==============================================================================
    print("\n🔹 RC Parameter Identification")
    id_data = []

    # ==============================================================================
    # Fit Rth/Cth
    # ==============================================================================
    for id_sel in id_list:
        # ------------------------------------------
        # Extract data
        # ------------------------------------------
        df_step = df[df["id"] == id_sel].copy().head(3500)
        time_step = df_step["time"].values - df_step["time"].values[0]
        T_amb = df_step[selR].values
        T_step = df_step[selY].values
        Is = df_step["Is"].values / np.sqrt(2)
        Wm = df_step["Wm"].values

        # ------------------------------------------
        # Scale losses
        # ------------------------------------------
        f1 = (1 + alpha * (T_step - Tref))
        f2 = 1 + beta_1 * (Wm / n_max) + beta_2 * (Wm / n_max) ** 2
        P_step = 3 * Rs * Is ** 2 * f1 * f2
        dt_s = np.mean(np.diff(time_step))

        # ------------------------------------------
        # Fit model
        # ------------------------------------------
        R_fit, C_fit = identify_rc(P_step.flatten(), T_step, T_amb, dt_s)
        id_data.append({"id": id_sel, "Is": np.mean(Is), "Wm": np.mean(Wm),
                        "Pv": np.max(P_step), "R": R_fit, "C": C_fit})

    # ==============================================================================
    # Average Rth and Cth
    # ==============================================================================
    df_ident = pd.DataFrame(id_data)
    # poly_R = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(df_ident[["Is", "Wm"]], df_ident["R"])
    # poly_C = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(df_ident[["Is", "Wm"]], df_ident["C"])
    R_hat, C_hat = df_ident["R"].mean(), df_ident["C"].mean()
    print(f"Identified Average: R_th = {R_hat:.4f} K/W, C_th = {C_hat:.2f} J/K")

    ###################################################################################################################
    # Preprocessing
    ###################################################################################################################
    # ==============================================================================
    # Init
    # ==============================================================================
    print("\n🔹 Data Preprocessing")

    # ==============================================================================
    # Filter Input
    # ==============================================================================
    df["Ta"] = (df["Ta"].rolling(window=W, center=True, min_periods=1).median())
    df["Tc"] = (df["Tc"].rolling(window=W, center=True, min_periods=1).median())

    # ==============================================================================
    # Scale losses
    # ==============================================================================
    f1 = (1 + alpha * (df[selR] - Tref))
    f2 = 1 + beta_1 * (df["Wm"] / n_max) + beta_2 * (df["Wm"] / n_max) ** 2
    df["Pv_s"] = 3 * Rs * (df["Is"] / np.sqrt(2)) ** 2 * f1 * f2

    # ==============================================================================
    # Split sets
    # ==============================================================================
    train_ids = [i for i in df["id"].unique() if i not in (test_ids + val_ids)]
    df_trainval = df[df["id"].isin(train_ids + val_ids)].copy()
    df_test = df[df["id"].isin(test_ids)].copy()

    X_mean, X_std = df_trainval[selX].mean(), df_trainval[selX].std() + 1e-8
    T_min, T_max = df_trainval[selY].min(), df_trainval[selY].max()
    t_max = df_trainval["time_id"].max()

    X_all, T_all, Tamb_all = normalize(df_trainval, selX, selY, selR, X_mean, X_std, T_max, T_min)
    t_norm = df_trainval["time_id"].values / t_max
    X_all = np.concatenate([X_all, t_norm[:, np.newaxis]], axis=-1)

    ids_all = df_trainval["id"].values
    P_all = df_trainval["Pv_s"].values
    T0_all = df_trainval["T0"].values
    time_all = df_trainval["time_id"].values

    # ==============================================================================
    # Time vector
    # ==============================================================================
    dt_torch = torch.tensor(dt_s, dtype=torch.float32, device=DEVICE)

    ###################################################################################################################
    # K-Fold Cross Validation
    ###################################################################################################################
    if TRAIN_MODEL:
        # ==============================================================================
        # K-Fold Validation
        # ==============================================================================
        if N_FOLDS > 1:
            # ------------------------------------------
            # Init
            # ------------------------------------------
            print(f"\n🔹 Starting {N_FOLDS}-Fold Cross Validation")
            kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
            fold_metrics = []
            n_features = X_all.shape[1]

            # ------------------------------------------
            # Iterate over Fold
            # ------------------------------------------
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_all)):
                # Init
                print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

                # Split data
                X_train, X_val = X_all[train_idx], X_all[val_idx]
                T_train, T_val = T_all[train_idx], T_all[val_idx]
                Tamb_train, Tamb_val = Tamb_all[train_idx], Tamb_all[val_idx]
                P_train, P_val = P_all[train_idx], P_all[val_idx]
                T0_train, T0_val = T0_all[train_idx], T0_all[val_idx]
                t_train, t_val = time_all[train_idx], time_all[val_idx]

                # Create data loader
                train_loader = prepare_loader(X_train, T_train, P_train, Tamb_train, t_train, T0_train,
                                              seq_len, stride, batch_size, DEVICE, ids_all[train_idx], shuffle=True)
                val_loader = prepare_loader(X_val, T_val, P_val, Tamb_val, t_val, T0_val,
                                            seq_len, stride, batch_size, DEVICE, ids_all[val_idx], shuffle=True)

                # Init model
                model = LSTM_PINN(input_dim=n_features, output_dim=1, hidden_dim=hidden_dim,
                                  num_layers=num_layers, dropout=dropout).to(DEVICE)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

                # Training
                model = train_one_fold(model, optimizer, scheduler, train_loader, val_loader, dt_torch,
                                       R_hat, C_hat, T_min, T_max, lambda_phys, lambda_init,
                                       epochs, patience, f"{MDL_NAME}_fold{fold}.pt")

                # Error and Output
                mse, mae, r2, max_err = evaluate_fold(model, val_loader, T_min, T_max)
                print(f"Fold {fold + 1}: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}, MAX={max_err:.4f}")
                fold_metrics.append([mse, mae, r2, max_err])

            # ------------------------------------------
            # Total Error
            # ------------------------------------------
            fold_metrics = np.array(fold_metrics)
            print("\n🔹 Cross-Validation Summary:")
            print(f"MSE → mean={fold_metrics[:, 0].mean():.4f}, std={fold_metrics[:, 0].std():.4f}")
            print(f"MAE → mean={fold_metrics[:, 1].mean():.4f}, std={fold_metrics[:, 1].std():.4f}")
            print(f"R²  → mean={fold_metrics[:, 2].mean():.4f}, std={fold_metrics[:, 2].std():.4f}")
            print(f"MAX → mean={fold_metrics[:, 3].mean():.4f}, std={fold_metrics[:, 3].std():.4f}")

        # ==============================================================================
        # Single-Fold Validation
        # ==============================================================================
        else:
            # ------------------------------------------
            # Init
            # ------------------------------------------
            print("\n🔹 Single Training Run (Fixed Train/Validation Split)")
            df_train = df[df["id"].isin(train_ids)].copy()
            df_val = df[df["id"].isin(val_ids)].copy()
            n_features = X_all.shape[1]

            # ------------------------------------------
            # Split data
            # ------------------------------------------
            X_train, T_train, Tamb_train = normalize(df_train, selX, selY, selR, X_mean, X_std, T_max, T_min)
            X_val, T_val, Tamb_val = normalize(df_val, selX, selY, selR, X_mean, X_std, T_max, T_min)
            t_train = df_train["time_id"].values / t_max
            t_val = df_val["time_id"].values / t_max
            X_train = np.concatenate([X_train, t_train[:, np.newaxis]], axis=-1)
            X_val = np.concatenate([X_val, t_val[:, np.newaxis]], axis=-1)

            # ------------------------------------------
            # Data Loader
            # ------------------------------------------
            train_loader = prepare_loader(X_train, T_train, df_train["Pv_s"].values,
                                          Tamb_train, df_train["time_id"].values, df_train["T0"].values,
                                          seq_len, stride, batch_size, DEVICE, df_train["id"].values, shuffle=True)
            val_loader = prepare_loader(X_val, T_val, df_val["Pv_s"].values,
                                        Tamb_val, df_val["time_id"].values, df_val["T0"].values,
                                        seq_len, stride, batch_size, DEVICE, df_val["id"].values, shuffle=False)

            # ------------------------------------------
            # Init Model
            # ------------------------------------------
            model = LSTM_PINN(input_dim=n_features, output_dim=1, hidden_dim=hidden_dim,
                              num_layers=num_layers, dropout=dropout).to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

            # ------------------------------------------
            # Training
            # ------------------------------------------
            model = train_one_fold(model, optimizer, scheduler, train_loader, val_loader, dt_torch,
                                   R_hat, C_hat, T_min, T_max, lambda_phys, lambda_init,
                                   epochs, patience, MDL_NAME)

            # ------------------------------------------
            # Error
            # ------------------------------------------
            mse, mae, r2, max_err = evaluate_fold(model, val_loader, T_min, T_max)
            print(f"\nSingle run results: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}, MAX={max_err:.4f}")

    ###################################################################################################################
    # Final Test Evaluation (Generalization)
    ###################################################################################################################
    # ==============================================================================
    # Init
    # ==============================================================================
    print("\n🔹 Evaluating Test Set")

    # ==============================================================================
    # Generate Test Sessions
    # ==============================================================================
    test_sessions = []
    for sid in test_ids:
        df_session = df_test[df_test["id"] == sid].copy()
        if df_session.empty:
            continue

        # Physics baseline
        P_test = df_session["Pv_s"].values
        T_pred_rc = foster_rc(df_session[selR].values, P_test, dt_s, R_hat, C_hat)

        # Prepare NN inputs and true temperatures (inverse-normalized later)
        X_test, T_test, _ = normalize(df_session, selX, selY, selR, X_mean, X_std, T_max, T_min)
        t_norm = df_session["time_id"].values / t_max
        X_test = np.concatenate([X_test, t_norm[:, np.newaxis]], axis=-1)

        # Inverse-normalized true temperature (store for global reference)
        T_true_phys = T_test * (T_max - T_min) + T_min

        # store
        test_sessions.append({
            "sid": sid,
            "df": df_session,
            "X_test": X_test,
            "T_true": T_true_phys,
            "time_min": np.linspace(0, (len(T_true_phys) - 1) / 60 * dt_s, len(T_true_phys)),
            "T_pred_rc": T_pred_rc
        })

    # ==============================================================================
    # Evaluate Test Sessions
    # ==============================================================================
    if not test_sessions:
        print("⚠️ No test sessions available (test_ids empty or not found).")
    else:
        T_true_global = np.concatenate([s["T_true"] for s in test_sessions], axis=0)

        # --- Create model instance once before loading weights ---
        model = LSTM_PINN(input_dim=X_test.shape[1], output_dim=1, hidden_dim=hidden_dim,
                          num_layers=num_layers, dropout=dropout).to(DEVICE)

        # Prepare container for fold predictions
        if N_FOLDS == 1:
            # Single-run: only one model file (MDL_NAME)
            model_path = MDL_NAME
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()

            T_pred_all = np.empty((0,))
            Tc_all = np.empty((0,))
            Is_all = np.empty((0,))
            Us_all = np.empty((0,))
            for s in test_sessions:
                X_test = s["X_test"]
                with torch.no_grad():
                    X_tensor = torch.tensor(X_test, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    T_pred_nn = model(X_tensor).cpu().numpy().reshape(-1)
                    T_pred_nn_phys = T_pred_nn * (T_max - T_min) + T_min

                T_pred_all = np.append(T_pred_all, T_pred_nn_phys)
                Tc_all = np.append(Tc_all, s["df"][selR])
                Is_all = np.append(Is_all, s["df"]["Is"])
                Us_all = np.append(Us_all, s["df"]["Us"])

                # Per-session metrics
                err_nn = T_pred_nn_phys - s["T_true"]
                mse_test = np.mean(err_nn ** 2)
                mae_test = np.mean(np.abs(err_nn))
                max_test = np.max(np.abs(err_nn))
                r2_test = r2_score(s["T_true"], T_pred_nn_phys)
                print(f"Session {s['sid']}: MSE={mse_test:.4f}, MAE={mae_test:.4f}, R2={r2_test:.4f}, MAX={max_test:.4f}")

            # Global metrics
            err_all = T_pred_all - T_true_global
            print("\n🔹 Overall Test Results:")
            print(f"MSE={np.mean(err_all ** 2):.4f}, MAE={np.mean(np.abs(err_all)):.4f}, "
                  f"R2={r2_score(T_true_global, T_pred_all):.4f}, MAX={np.max(np.abs(err_all)):.4f}")

            # Plot measured vs predicted
            if ENABLE_PLOTS:
                time = np.linspace(0, (len(err_all) - 1) / 60 * dt_s, len(err_all))  # time in minutes
                fig, axs = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
                fig.suptitle(f"Model Comparison", fontsize=12, fontweight="bold")

                # ▪️ Top: Current Voltage
                axs[0].plot(time, Is_all, label="Current", linewidth=2)
                axs[0].plot(time, Us_all, label="Voltage", linewidth=2)
                axs[0].set_ylabel("Current [A] / Voltage [V]")
                axs[0].legend(loc="best")
                axs[0].grid(True, linestyle="--", linewidth=0.6)

                # ▪️ Mid: Temperatures
                axs[1].plot(time, T_true_global, label="True", color="black", linewidth=2)
                axs[1].plot(time, T_pred_all, label="Pred", linewidth=2)
                axs[1].plot(time, Tc_all, label="Coolant", linewidth=2)
                axs[1].set_ylabel("Temperature [°C]")
                axs[1].legend(loc="best")
                axs[1].grid(True, linestyle="--", linewidth=0.6)

                # ▪️ Bottom: Errors
                axs[2].plot(time, err_all, label="Error")
                axs[2].axhline(0, color="black", linewidth=1)
                axs[2].set_xlabel("Time [min]")
                axs[2].set_ylabel("Error [°C]")
                axs[2].legend(loc="best")
                axs[2].grid(True, linestyle="--", linewidth=0.6)

                plt.tight_layout()
                plt.show()

        else:
            # k-fold: evaluate each fold model on the same test set and aggregate
            T_pred_folds = []  # will become array of shape (n_folds, N_total)
            fold_metrics = []

            for fold in range(N_FOLDS):
                model_path = f"{MDL_NAME}_fold{fold}.pt"
                if not os.path.exists(model_path):
                    print(f"⚠️ Model for fold {fold} not found: {model_path} (skipping)")
                    continue

                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.to(DEVICE)
                model.eval()

                T_pred_concat = np.empty((0,))

                for s in test_sessions:
                    X_test = s["X_test"]
                    with torch.no_grad():
                        X_tensor = torch.tensor(X_test, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        T_pred_nn = model(X_tensor).cpu().numpy().reshape(-1)
                        T_pred_nn_phys = T_pred_nn * (T_max - T_min) + T_min

                    T_pred_concat = np.append(T_pred_concat, T_pred_nn_phys)

                # store predictions for this fold
                T_pred_folds.append(T_pred_concat)

                # compute metrics for this fold (concatenated)
                err = T_pred_concat - T_true_global
                mse_f = np.mean(err ** 2)
                mae_f = np.mean(np.abs(err))
                r2_f = r2_score(T_true_global, T_pred_concat)
                max_f = np.max(np.abs(err))
                fold_metrics.append({"fold": fold, "MSE": mse_f, "MAE": mae_f, "R2": r2_f, "MAX": max_f})
                print(f"Fold {fold}: MSE={mse_f:.4f}, MAE={mae_f:.4f}, R2={r2_f:.4f}, MAX={max_f:.4f}")

            # convert to array [n_valid_folds, N_total]
            if len(T_pred_folds) == 0:
                raise RuntimeError("No fold predictions were collected (check model files).")
            T_pred_folds = np.vstack(T_pred_folds)  # shape (n_folds_actual, N_total)

            # compute pointwise mean & std across folds
            T_pred_mean = np.mean(T_pred_folds, axis=0)
            T_pred_std = np.std(T_pred_folds, axis=0)

            # aggregate metrics across folds (mean ± std)
            metrics_df = pd.DataFrame(fold_metrics).set_index("fold")
            metrics_mean = metrics_df.mean()
            metrics_std = metrics_df.std()

            print("\n🔹 Overall Test Results (across folds):")
            print(f"MSE = {metrics_mean['MSE']:.4f} ± {metrics_std['MSE']:.4f}")
            print(f"MAE = {metrics_mean['MAE']:.4f} ± {metrics_std['MAE']:.4f}")
            print(f"R2  = {metrics_mean['R2']:.4f} ± {metrics_std['R2']:.4f}")
            print(f"MAX = {metrics_mean['MAX']:.4f} ± {metrics_std['MAX']:.4f}")

            # Plot averaged prediction with ±1σ band
            if ENABLE_PLOTS:
                time = np.linspace(0, len(T_true_global) / 60 * dt_s, len(T_true_global))
                fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                fig.suptitle(f"Model Comparison", fontsize=12, fontweight="bold")

                # ▪️ Mid: Temperatures
                axs[0].plot(time, T_true_global, label="True", linewidth=2)
                axs[0].plot(time, T_pred_mean, label="Pred", linewidth=1.8)
                axs[0].fill_between(time, T_pred_mean - T_pred_std, T_pred_mean + T_pred_std,
                                    alpha=0.25, label="±1σ across folds")
                axs[0].set_title(f"Test Set Prediction (Mean ± Std) — {len(T_pred_folds)} folds evaluated")
                axs[0].set_xlabel("Time [min]")
                axs[0].set_ylabel("Temperature [°C]")
                axs[0].legend()
                axs[0].grid(True, linestyle="--", linewidth=0.6)

                # ▪️ Bottom: Errors
                axs[1].plot(time, T_true_global-T_pred_mean, label="Error")
                axs[1].fill_between(time, (T_true_global-T_pred_mean) - T_pred_std, (T_true_global-T_pred_mean) + T_pred_std,
                                    alpha=0.25, label="±1σ across folds")
                axs[1].axhline(0, color="black", linewidth=1)
                axs[1].set_xlabel("Time [min]")
                axs[1].set_ylabel("Error [°C]")
                axs[1].legend(loc="best")
                axs[1].grid(True, linestyle="--", linewidth=0.6)

                plt.tight_layout()
                plt.show()


#######################################################################################################################
if __name__ == "__main__":
    main()
