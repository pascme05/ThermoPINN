from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.model import *


#######################################################################################################################
# Helper: Train one fold
#######################################################################################################################
def train_one_fold(model, optimizer, scheduler, train_loader, val_loader, dt_torch,
                   R_hat, C_hat, N_nodes, T_min, T_max, lambda_phys, lambda_init,
                   epochs, patience, mdl_name):
    best_val_loss = np.inf
    patience_counter = 0

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for Xb, Tb, Pb, Tambb, t_seq, T0b in train_loader:
            optimizer.zero_grad()
            loss, d_mse, p_mse, i_mse = pinn_loss_lstm(
                model, Xb, Tb, Pb, t_seq, T0b, dt_torch,
                R_hat, C_hat, N_nodes, Tambb, T_min, T_max,
                lambda_phys, lambda_init
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        val_total, val_data, val_phys, val_init = 0, 0, 0, 0
        with torch.no_grad():
            for Xv, Tv, Pv, Tambv, tv_seq, T0v in val_loader:
                v_loss, v_dmse, v_pmse, v_imse = pinn_loss_lstm(
                    model, Xv, Tv, Pv, tv_seq, T0v, dt_torch,
                    R_hat, C_hat, N_nodes, Tambv, T_min, T_max,
                    lambda_phys, lambda_init
                )
                val_total += v_loss.item()
                val_data += v_dmse
                val_phys += v_pmse
                val_init += v_imse

        train_loss = total_loss / len(train_loader)
        val_loss = val_total / len(val_loader)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)

        print(f"Epoch {ep+1:03d} | LR={current_lr:.6f} | "
                  f"Train={train_loss:.6f} | Val={val_loss:.6f} | "
                  f"Data={val_data/len(val_loader):.6f} | Phys={val_phys/len(val_loader):.6f} | "
                  f"Init={val_init/len(val_loader):.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), mdl_name)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {ep+1}")
                break

    model.load_state_dict(torch.load(mdl_name))
    return model


#######################################################################################################################
# Helper: Evaluate one fold
#######################################################################################################################
def evaluate_fold(model, val_loader, T_min, T_max):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for Xv, Tv, Pv, Tambv, tv_seq, T0v in val_loader:
            T_pred = model(Xv).squeeze().cpu().numpy()
            T_true = Tv.cpu().numpy()

            # Inverse normalization
            T_pred = T_pred * (T_max - T_min) + T_min
            T_true = T_true * (T_max - T_min) + T_min

            y_true.extend(T_true)
            y_pred.extend(T_pred)

    # Concatenate safely
    y_pred = np.concatenate([np.ravel(p) for p in y_pred])
    y_true = np.concatenate([np.ravel(t) for t in y_true])

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    max_err = np.max(np.abs(y_true - y_pred))
    return mse, mae, r2, max_err

