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
from sklearn.model_selection import GroupKFold
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import LinearRegression
# from sklearn.pipeline import make_pipeline
from src.helpTrain import *
from scipy.stats import norm
from torchinfo import summary


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
    TRAIN_MODEL = True                                                                                                   # True: Model will be trained and tested, False: Testing
    ENABLE_PLOTS = True                                                                                                  # True: Plot results
    ORIGINAL_DATA = False                                                                                                # True. Transforms the original Kaggle data into the correct format
    FIT_RC = True                                                                                                        # True: Fits the RC parameters based on measured step responses
    MDL = 2                                                                                                              # 0) LSTM, 1) Hidden-State LSTM, 2) LSTM Warmup
    N_FOLDS = 1                                                                                                          # Number of cross-validation runs

    # ==============================================================================
    # Paths
    # ==============================================================================
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "data", "motor_temp.csv")
    if ORIGINAL_DATA:
        DATA_PATH = os.path.join(BASE_DIR, "data", "measures_v2.csv")
    MDL_NAME = os.path.join(BASE_DIR, "mdl", "mdl_multi_pinn_1Hz.pt")

    # ==============================================================================
    # General Parameter
    # ==============================================================================
    W = 5                                                                                                                # Window length for filtering data

    # ==============================================================================
    # Physical parameters
    # ==============================================================================
    # ------------------------------------------
    # Losses
    # ------------------------------------------
    Rs = 14.1e-3                                                                                                         # Stator resistance [Ohm]
    alpha = 0.00393                                                                                                      # Temperature coefficient of resistance [1/°C]
    Tref = 20                                                                                                            # Reference temperature for Rs [°C]
    n_max = 6000                                                                                                         # Maximum motor speed [rpm]
    beta_1 = 0.315                                                                                                       # Parameter for frequency losses (linear)
    beta_2 = 0.616                                                                                                       # Parameter for frequency losses (quadratic)

    # ------------------------------------------
    # Thermal
    # ------------------------------------------
    N_nodes = 1                                                                                                          # Number of RC nodes
    Rth = [0.0057, 0.0353]                                                                                               # List of thermal resistances [K/W]
    tau = [596, 222]                                                                                                     # List of thermal time constants [sec]

    # ==============================================================================
    # Training hyperparameters
    # ==============================================================================
    # ------------------------------------------
    # Opti Para
    # ------------------------------------------
    """
    seq_len = 1400  # Sequence length (timesteps per training sample)
    stride = 30  # Step size between training sequences
    batch_size = 32  # Batch size for training
    hidden_dim = 256  # Hidden units in LSTM layers
    num_layers = 2  # Number of stacked LSTM layers
    dropout = 0.25  # Dropout for the LSTM layer
    lr = 1.67e-3  # Learning rate for optimizer
    epochs = 100  # Maximum number of training epochs
    lambda_phys = 0.1  # Weight for physics-informed loss term
    lambda_init = 0.5  # Weight for initial condition loss
    patience = 10  # Early stopping patience (epochs without improvement)
    """

    # ------------------------------------------
    # Test Para
    # ------------------------------------------
    seq_len = 1400                                                                                                       # Sequence length (timesteps per training sample)
    stride = 30                                                                                                         # Step size between training sequences
    batch_size = 32                                                                                                      # Batch size for training
    hidden_dim = 256                                                                                                      # Hidden units in LSTM layers
    num_layers = 2                                                                                                       # Number of stacked LSTM layers
    dropout = 0.25                                                                                                       # Dropout for the LSTM layer
    lr = 1.67e-3                                                                                                         # Learning rate for optimizer
    epochs = 10                                                                                                          # Maximum number of training epochs
    lambda_phys = 0.1                                                                                                    # Weight for physics-informed loss term
    lambda_init = 0.5                                                                                                    # Weight for initial condition loss
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
    selX = ["Ta", "Tc", "Is", "Us", "Wm"]
    selY = "Tsw"
    selYM = ["Tsw", "Tso", "Tst", "Trm"]
    selR = "Tc"

    ###################################################################################################################
    # Load data
    ###################################################################################################################
    # ==============================================================================
    # Input and Output Select
    # ==============================================================================
    df = pd.read_csv(DATA_PATH)

    # ==============================================================================
    # Original data
    # ==============================================================================
    if ORIGINAL_DATA:
        # ------------------------------------------
        # Rename
        # ------------------------------------------
        df = df.rename(columns={'ambient': 'Ta', 'coolant': 'Tc', 'i_d': 'Id', 'i_q': 'Iq', 'u_d': 'Ud', 'u_q': 'Uq',
                                'profile_id': 'id', 'torque': 'Mm', 'motor_speed': 'Wm', 'stator_winding': 'Tsw'})

        # ------------------------------------------
        # Calculate features
        # ------------------------------------------
        df["Is"] = (df["Id"] ** 2 + df["Iq"] ** 2) ** 0.5
        df["Us"] = (df["Ud"] ** 2 + df["Uq"] ** 2) ** 0.5
        df['T0'] = df.groupby('id')['Tsw'].transform('first')
        df["time"] = np.linspace(0, (len(df["Is"])*0.5 - 1/2), len(df["Is"]))
        df['time_id'] = df.groupby('id')['time'].transform(lambda x: x - x.iloc[0])
    df["Sel"] = 3 / 2 * df["Is"] * df["Us"]
    df["SelI"] = df["Sel"] * df["Is"]
    df["SelW"] = df["Sel"] * df["Wm"]

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
    if FIT_RC:
        for id_sel in id_list:
            # ------------------------------------------
            # Extract data
            # ------------------------------------------
            df_step = df[df["id"] == id_sel].copy().head(3500)
            time_step = df_step["time"].values - df_step["time"].values[0]
            T_amb = df_step[selR].values
            T_step = df_step[selY].values
            dT = T_step - T_step[0]
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
            if N_nodes > 1:
                R_fit, C_fit = fit_zth(time_step, dT, P_step.mean(), num_nodes=N_nodes)
                print(f"Identified: R_th = {np.sum(R_fit):.4f} K/W, tau = {np.sum(C_fit*R_fit):.1f} sec")
            else:
                R_fit, C_fit = identify_rc(P_step.flatten(), T_step, T_amb, dt_s)
            id_data.append({"id": id_sel, "Is": np.mean(Is), "Wm": np.mean(Wm), "Pv": np.max(P_step),
                            "R": R_fit, "C": C_fit})

        # ------------------------------------------
        # Average Rth and Cth
        # ------------------------------------------
        df_ident = pd.DataFrame(id_data)
        # poly_R = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(df_ident[["Is", "Wm"]], df_ident["R"])
        # poly_C = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(df_ident[["Is", "Wm"]], df_ident["C"])
        R_hat, C_hat, tau = df_ident["R"].mean(), df_ident["C"].mean(), df_ident["R"].mean()*df_ident["C"].mean()
        print(f"Identified Average: R_th_sum = {np.sum(R_hat):.4f} K/W, C_th_min = {np.min(C_hat):.2f} J/K, and tau_min = {np.min(tau):.2f} sec")

    # ==============================================================================
    # Set RC
    # ==============================================================================
    else:
        # ------------------------------------------
        # Average Rth and Cth
        # ------------------------------------------
        R_hat = np.array(Rth, dtype='float32')
        tau = np.array(tau, dtype='float32')
        C_hat = tau / Rth
        N_nodes = len(R_hat)
        print(f"Defined Average: R_th_sum = {np.sum(R_hat):.4f} K/W, C_th_min = {np.min(C_hat):.2f} J/K, and tau_min = {np.min(tau):.2f} sec")

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
    T_min_all, T_max_all = df_trainval[selYM].min().values, df_trainval[selYM].max().values
    T_min, T_max = df_trainval[selY].min(), df_trainval[selY].max()
    t_max = df_trainval["time_id"].max()

    X_all, T_all, Tamb_all = normalize(df_trainval, selX, selYM, selR, X_mean, X_std, T_max, T_min, T_max_all, T_min_all)
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
            kfold = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
            fold_metrics = []
            n_features = X_all.shape[1]

            # ------------------------------------------
            # Iterate over Fold
            # ------------------------------------------
            for fold, (train_idx, val_idx) in enumerate(kfold.split(X_all, y=None, groups=df_trainval["id"].values)):
                # Init
                print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")
                print(f"Fold {fold + 1}: {len(np.unique(X_all[train_idx]))} train IDs, {len(np.unique(X_all[val_idx]))} val IDs")

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
                if MDL == 1:
                    model = LSTM_PINN_HIDDEN(input_dim=n_features, output_dim=len(selYM), hidden_dim=hidden_dim,
                                             num_layers=num_layers, dropout=dropout).to(DEVICE)
                elif MDL == 2:
                    model = LSTM_PINN_WARM(input_dim=n_features, output_dim=len(selYM), hidden_dim=hidden_dim,
                                           num_layers=num_layers, dropout=dropout).to(DEVICE)
                else:
                    model = LSTM_PINN(input_dim=n_features, output_dim=len(selYM), hidden_dim=hidden_dim,
                                      num_layers=num_layers, dropout=dropout).to(DEVICE)
                summary(model, input_size=(batch_size, seq_len, n_features))

                # Optimizer
                optimizer = optim.Adam(model.parameters(), lr=lr)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

                # Training
                model = train_one_fold_multi(model, optimizer, scheduler, train_loader, val_loader, dt_torch,
                                             R_hat, C_hat, N_nodes, T_min, T_max, lambda_phys, lambda_init,
                                             epochs, patience, f"{MDL_NAME}_fold{fold}.pt")

                # Error and Output
                mse, mae, r2, max_err = evaluate_fold(model, val_loader, T_min_all, T_max_all)
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
            X_train, T_train, Tamb_train = normalize(df_train, selX, selYM, selR, X_mean, X_std, T_max, T_min, T_max_all, T_min_all)
            X_val, T_val, Tamb_val = normalize(df_val, selX, selYM, selR, X_mean, X_std, T_max, T_min, T_max_all, T_min_all)
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
            if MDL == 1:
                model = LSTM_PINN_HIDDEN(input_dim=n_features, output_dim=len(selYM), hidden_dim=hidden_dim,
                                         num_layers=num_layers, dropout=dropout).to(DEVICE)
            elif MDL == 2:
                model = LSTM_PINN_WARM(input_dim=n_features, output_dim=len(selYM), hidden_dim=hidden_dim,
                                       num_layers=num_layers, dropout=dropout).to(DEVICE)
            else:
                model = LSTM_PINN(input_dim=n_features, output_dim=len(selYM), hidden_dim=hidden_dim,
                                  num_layers=num_layers, dropout=dropout).to(DEVICE)
            summary(model, input_size=(batch_size, seq_len, n_features))

            # ------------------------------------------
            # Init Opti
            # ------------------------------------------
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)

            # ------------------------------------------
            # Training
            # ------------------------------------------
            model = train_one_fold_multi(model, optimizer, scheduler, train_loader, val_loader, dt_torch,
                                         R_hat, C_hat, N_nodes, T_min, T_max, lambda_phys, lambda_init,
                                         epochs, patience, MDL_NAME)

            # ------------------------------------------
            # Error
            # ------------------------------------------
            mse, mae, r2, max_err = evaluate_fold(model, val_loader, T_min_all, T_max_all)
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
    X_test = []
    for sid in test_ids:
        df_session = df_test[df_test["id"] == sid].copy()
        if df_session.empty:
            continue

        # Physics baseline
        # P_test = df_session["Pv_s"].values
        # time = np.linspace(0, (len(P_test) - 1) * dt_s, len(P_test))
        # T_pred_rc = predict_rc(P_test, time, df_session[selR].values, R_hat, C_hat)
        # T_pred_rc = foster_rc(df_session[selR].values, P_test, dt_s, R_hat, C_hat)

        # Prepare NN inputs and true temperatures (inverse-normalized later)
        X_test, T_test, _ = normalize(df_session, selX, selYM, selR, X_mean, X_std, T_max, T_min, T_max_all, T_min_all)
        t_norm = df_session["time_id"].values / t_max
        X_test = np.concatenate([X_test, t_norm[:, np.newaxis]], axis=-1)

        # Inverse-normalized true temperature (store for global reference)
        T_true_phys = T_test * (T_max_all - T_min_all) + T_min_all

        # store
        test_sessions.append({
            "sid": sid,
            "df": df_session,
            "X_test": X_test,
            "T_true": T_true_phys,
            "time_min": np.linspace(0, (len(T_true_phys) - 1) / 60 * dt_s, len(T_true_phys))
        })

    # ==============================================================================
    # Evaluate Test Sessions
    # ==============================================================================
    if not test_sessions:
        print("⚠️ No test sessions available (test_ids empty or not found).")
    else:
        T_true_global = np.concatenate([s["T_true"] for s in test_sessions], axis=0)

        # --- Create model instance once before loading weights ---
        if MDL == 1:
            model = LSTM_PINN_HIDDEN(input_dim=X_test.shape[1], output_dim=len(selYM), hidden_dim=hidden_dim,
                                     num_layers=num_layers, dropout=dropout).to(DEVICE)
        elif MDL == 2:
            model = LSTM_PINN_WARM(input_dim=X_test.shape[1], output_dim=len(selYM), hidden_dim=hidden_dim,
                                   num_layers=num_layers, dropout=dropout).to(DEVICE)
        else:
            model = LSTM_PINN(input_dim=X_test.shape[1], output_dim=len(selYM), hidden_dim=hidden_dim,
                              num_layers=num_layers, dropout=dropout).to(DEVICE)
        summary(model, input_size=(batch_size, seq_len, X_test.shape[1]))

        # Prepare container for fold predictions
        if N_FOLDS == 1:
            # Single-run: only one model file (MDL_NAME)
            model_path = MDL_NAME
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()

            T_pred_all = np.empty((0, len(selYM)))
            Tc_all = np.empty((0,))
            Is_all = np.empty((0,))
            Us_all = np.empty((0,))
            for s in test_sessions:
                X_test = s["X_test"]
                with torch.no_grad():
                    X_tensor = torch.tensor(X_test, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                    T_pred_nn = np.squeeze(model(X_tensor).cpu().numpy(), axis=0)
                    T_pred_nn_phys = T_pred_nn * (T_max_all - T_min_all) + T_min_all

                T_pred_all = np.vstack([T_pred_all, T_pred_nn_phys])
                Tc_all = np.append(Tc_all, s["df"][selR])
                Is_all = np.append(Is_all, s["df"]["Is"])
                Us_all = np.append(Us_all, s["df"]["Us"])

                # Per-session metrics
                err_nn = T_pred_nn_phys - s["T_true"]
                mse_test = np.mean(err_nn ** 2, axis=0)
                mae_test = np.mean(np.abs(err_nn), axis=0)
                max_test = np.max(np.abs(err_nn), axis=0)
                for i in range(0, len(selYM)):
                    print(f"Session {s['sid']} and Node {i}: MSE={mse_test[i]:.4f}, MAE={mae_test[i]:.4f}, MAX={max_test[i]:.4f}")

            # Global metrics
            err_all = T_pred_all - T_true_global
            print("\n🔹 Overall Test Results:")
            for i in range(0, len(selYM)):
                print(f"Node {i}: MSE={np.mean(err_all[:, i] ** 2):.4f}, MAE={np.mean(np.abs(err_all[:, i])):.4f}, MAX={np.max(np.abs(err_all[:, i])):.4f}")

            # Plot measured vs predicted
            if ENABLE_PLOTS:
                for i in range(0, len(selYM)):
                    prop_cycle = plt.rcParams['axes.prop_cycle']
                    colors = prop_cycle.by_key()['color']

                    time = np.linspace(0, (len(err_all) - 1) / 60 * dt_s, len(err_all))
                    fig, axs = plt.subplots(3, 2, gridspec_kw={'width_ratios': [3, 1]}, figsize=(10, 6), sharex=False)
                    fig.suptitle(f"Model Comparison Node {i}", fontsize=12, fontweight="bold")

                    # ▪️ Top: Current Voltage
                    axs[0, 0].plot(time, Is_all, label="Current", linewidth=2)
                    axs[0, 0].plot(time, Us_all, label="Voltage", linewidth=2)
                    axs[0, 0].set_ylabel("Current [A] / Voltage [V]")
                    axs[0, 0].legend(loc="best")
                    axs[0, 0].grid(True, linestyle="--", linewidth=0.6)

                    mu, std = norm.fit(Is_all[Is_all>20])
                    axs[0, 1].hist(Is_all[Is_all>20], bins=50, label="RMS Current", density=True, alpha=0.6)
                    x_prob = np.linspace(0, 1.2*Is_all.max(), 100)
                    p = norm.pdf(x_prob, mu, std)
                    axs[0, 1].plot(x_prob, p, linewidth=2, color=colors[0])

                    mu, std = norm.fit(Us_all[Us_all>20])
                    axs[0, 1].hist(Us_all[Us_all>20], bins=50, label="RMS Voltage", density=True, alpha=0.6)
                    x_prob = np.linspace(0.8*Us_all.min(), 1.2*Us_all.max(), 100)
                    p = norm.pdf(x_prob, mu, std)
                    axs[0, 1].plot(x_prob, p, linewidth=2, color=colors[1])
                    axs[0, 1].set_xlabel("Current [A] / Voltage [V]")
                    axs[0, 1].set_ylabel("Probability [%]")
                    axs[0, 1].legend(loc="best")
                    axs[0, 1].grid(True, linestyle="--", linewidth=0.6)

                    # ▪️ Mid: Temperatures
                    axs[1, 0].plot(time, T_true_global[:, i], label="True", color="black", linewidth=2)
                    axs[1, 0].plot(time, T_pred_all[:, i], label="Pred", linewidth=2)
                    axs[1, 0].plot(time, Tc_all, label="Coolant", linewidth=2)
                    axs[1, 0].set_ylabel("Temperature [°C]")
                    axs[1, 0].legend(loc="best")
                    axs[1, 0].grid(True, linestyle="--", linewidth=0.6)

                    mu, std = norm.fit(T_true_global[:, i])
                    axs[1, 1].hist(T_true_global[:, i], bins=50, label="True", density=True, alpha=0.6)
                    x_prob = np.linspace(0.8*T_true_global[:, i].min(), 1.2*T_true_global[:, i].max(), 100)
                    p = norm.pdf(x_prob, mu, std)
                    axs[1, 1].plot(x_prob, p, linewidth=2, color=colors[0])

                    mu, std = norm.fit(T_pred_all[:, i])
                    axs[1, 1].hist(T_pred_all[:, i], bins=50, label="Pred", density=True, alpha=0.6)
                    x_prob = np.linspace(0.8*T_pred_all[:, i].min(), 1.2*T_pred_all[:, i].max(), 100)
                    p = norm.pdf(x_prob, mu, std)
                    axs[1, 1].plot(x_prob, p, linewidth=2, color=colors[1])
                    axs[1, 1].set_xlabel("Temperature [°C]")
                    axs[1, 1].set_ylabel("Probability [%]")
                    axs[1, 1].legend(loc="best")
                    axs[1, 1].grid(True, linestyle="--", linewidth=0.6)

                    # ▪️ Bottom: Errors
                    axs[2, 0].plot(time, err_all[:, i], label="Error")
                    axs[2, 0].axhline(0, color="black", linewidth=1)
                    axs[2, 0].set_xlabel("Time [min]")
                    axs[2, 0].set_ylabel("Error [°C]")
                    axs[2, 0].legend(loc="best")
                    axs[2, 0].grid(True, linestyle="--", linewidth=0.6)

                    mu, std = norm.fit(err_all[:, i])
                    axs[2, 1].hist(err_all[:, i], bins=100, label="Error", density=True, alpha=0.6)
                    x_prob = np.linspace(err_all[:, i].min(), err_all[:, i].max(), 100)
                    p = norm.pdf(x_prob, mu, std)
                    axs[2, 1].plot(x_prob, p, linewidth=2, color=colors[0])
                    axs[2, 1].set_xlabel("Error [K]")
                    axs[2, 1].set_ylabel("Probability [%]")
                    axs[2, 1].legend(loc="best")
                    axs[2, 1].grid(True, linestyle="--", linewidth=0.6)

                    plt.tight_layout()
                plt.show()

        else:
            # k-fold: evaluate each fold model on the same test set and aggregate
            T_pred_folds = []
            fold_metrics = []

            for fold in range(N_FOLDS):
                model_path = f"{MDL_NAME}_fold{fold}.pt"
                if not os.path.exists(model_path):
                    print(f"⚠️ Model for fold {fold} not found: {model_path} (skipping)")
                    continue

                model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                model.to(DEVICE)
                model.eval()

                # Determine output feature dimension K dynamically from first prediction
                first_pred = None
                for s in test_sessions:
                    X_test = s["X_test"]
                    with torch.no_grad():
                        X_tensor = torch.tensor(X_test, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        pred = model(X_tensor).cpu().numpy()
                        if first_pred is None:
                            first_pred = pred
                            K = first_pred.shape[-1]  # assumes output shape (M, K)
                    break  # only need first sample to get K

                # Initialize empty array with shape (0, K) to stack predictions vertically
                T_pred_concat = np.empty((0, K))

                # Process all test sessions
                for s in test_sessions:
                    X_test = s["X_test"]
                    with torch.no_grad():
                        X_tensor = torch.tensor(X_test, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                        T_pred_nn = model(X_tensor).cpu().numpy().reshape(-1, K)  # shape (M, K)
                        T_pred_nn_phys = T_pred_nn * (T_max_all - T_min_all) + T_min_all  # apply scaling per feature

                    T_pred_concat = np.vstack([T_pred_concat, T_pred_nn_phys])

                T_pred_folds.append(T_pred_concat)

                # Calculate errors and metrics per feature
                err = T_pred_concat - T_true_global  # shape (N_total, K)

                mse_f = np.mean(err ** 2, axis=0)
                mae_f = np.mean(np.abs(err), axis=0)
                r2_f = np.array([r2_score(T_true_global[:, i], T_pred_concat[:, i]) for i in range(K)])
                max_f = np.max(np.abs(err), axis=0)

                metrics_entry = {"fold": fold}
                for i in range(K):
                    metrics_entry.update({
                        f"MSE_{i + 1}": mse_f[i],
                        f"MAE_{i + 1}": mae_f[i],
                        f"R2_{i + 1}": r2_f[i],
                        f"MAX_{i + 1}": max_f[i]})
                fold_metrics.append(metrics_entry)

                for i in range(0, len(selYM)):
                    print(f"Fold {fold} and Node {i}: MSE={mse_f[i]:.4f}, MAE={mae_f[i]:.4f}, MAX={max_f[i]:.4f}")

            if len(T_pred_folds) == 0:
                raise RuntimeError("No fold predictions were collected (check model files).")

            T_pred_folds = np.stack(T_pred_folds, axis=0)  # shape (n_folds_actual, N_total, K)

            # Compute pointwise mean & std across folds
            T_pred_mean = np.mean(T_pred_folds, axis=0)  # shape (N_total, K)
            T_pred_std = np.std(T_pred_folds, axis=0)  # shape (N_total, K)

            metrics_df = pd.DataFrame(fold_metrics).set_index("fold")

            metrics_mean = metrics_df.mean()
            metrics_std = metrics_df.std()

            print("\n🔹 Overall Test Results (across folds):")
            for i in range(K):
                print(f"Node {i + 1}:")
                print(f"  MSE = {metrics_mean[f'MSE_{i + 1}']:.4f} ± {metrics_std[f'MSE_{i + 1}']:.4f}")
                print(f"  MAE = {metrics_mean[f'MAE_{i + 1}']:.4f} ± {metrics_std[f'MAE_{i + 1}']:.4f}")
                print(f"  R2  = {metrics_mean[f'R2_{i + 1}']:.4f} ± {metrics_std[f'R2_{i + 1}']:.4f}")
                print(f"  MAX = {metrics_mean[f'MAX_{i + 1}']:.4f} ± {metrics_std[f'MAX_{i + 1}']:.4f}")

            # Plot averaged prediction with ±1σ band
            if ENABLE_PLOTS:
                for i in range(0, len(selYM)):
                    time = np.linspace(0, len(T_true_global) / 60 * dt_s, len(T_true_global))
                    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                    fig.suptitle(f"Model Comparison Node {i}", fontsize=12, fontweight="bold")

                    # ▪️ Mid: Temperatures
                    axs[0].plot(time, T_true_global[:, i], label="True", linewidth=2)
                    axs[0].plot(time, T_pred_mean[:, i], label="Pred", linewidth=1.8)
                    axs[0].fill_between(time, T_pred_mean[:, i] - T_pred_std[:, i], T_pred_mean[:, i] + T_pred_std[:, i],
                                        alpha=0.25, label="±1σ across folds")
                    axs[0].set_title(f"Test Set Prediction (Mean ± Std) — {len(T_pred_folds)} folds evaluated")
                    axs[0].set_xlabel("Time [min]")
                    axs[0].set_ylabel("Temperature [°C]")
                    axs[0].legend()
                    axs[0].grid(True, linestyle="--", linewidth=0.6)

                    # ▪️ Bottom: Errors
                    axs[1].plot(time, T_true_global[:, i]-T_pred_mean[:, i], label="Error")
                    axs[1].fill_between(time, (T_true_global[:, i]-T_pred_mean[:, i]) - T_pred_std[:, i],
                                        (T_true_global[:, i]-T_pred_mean[:, i]) + T_pred_std[:, i],
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
