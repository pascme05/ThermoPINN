import torch.nn as nn
from src.auxiliary import *


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