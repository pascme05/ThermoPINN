import torch
import torch.nn as nn
from src.auxiliary import *


# ----------------------------------------------------
# LSTM-PINN Model
# ----------------------------------------------------
class LSTM_PINN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out.squeeze(-1)


# ----------------------------------------------------
# LSTM-PINN HIDDEN Model
# ----------------------------------------------------
class LSTM_PINN_HIDDEN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.init_state_net = nn.Linear(input_dim, hidden_dim)

    def forward(self, x):
        # Learn hidden init from first input
        h0 = torch.tanh(self.init_state_net(x[:, 0, :]))
        h0 = h0.unsqueeze(0).repeat(self.lstm.num_layers, 1, 1)
        c0 = torch.zeros_like(h0)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        # Residual to anchor early steps
        return out.squeeze(-1) + x[..., 0]


# ----------------------------------------------------
# LSTM-PINN Multi Model
# ----------------------------------------------------
class MultiPathLSTM_PINN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, output_dim: int = 1):
        """
        Multi-path LSTM architecture for time-series prediction.
        Input:  [B, W, F]  (Batch, Sequence length, Features)
        Output: [B, W, N]  if N > 1, else [B, W]
        """
        super().__init__()

        # --- 1️⃣ Time-domain path ---
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )

        # --- 2️⃣ Frequency-domain path ---
        self.freq_conv = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(hidden_dim)
        )

        # --- 3️⃣ Statistical path ---
        self.stat_mlp = nn.Sequential(
            nn.Linear(input_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # --- 4️⃣ Learnable weighting between paths ---
        self.alpha = nn.Parameter(torch.ones(3))  # 3-path weights

        # --- 5️⃣ Final projection to output_dim ---
        self.fc_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: [B, W, F]
        Returns:
            [B, W, N] if N > 1
            [B, W]    if N == 1
        """
        B, W, F = x.shape

        # --- (1) Time-domain path ---
        lstm_out, _ = self.lstm(x)  # [B, W, H]

        # --- (2) Frequency-domain path ---
        freq_repr = torch.fft.rfft(x, dim=1)             # [B, Wf, F]
        freq_mag = torch.abs(freq_repr).transpose(1, 2)  # [B, F, Wf] for Conv1d
        freq_out = self.freq_conv(freq_mag)              # [B, 32, H]
        freq_out = freq_out.mean(dim=1)                  # [B, H]
        freq_out = freq_out.unsqueeze(1).expand(-1, W, -1)  # [B, W, H]

        # --- (3) Statistical path ---
        mean = x.mean(dim=1)
        std = x.std(dim=1)
        skew = ((x - mean.unsqueeze(1))**3).mean(dim=1) / (std**3 + 1e-6)
        kurt = ((x - mean.unsqueeze(1))**4).mean(dim=1) / (std**4 + 1e-6)
        stats = torch.cat([mean, std, skew, kurt], dim=-1)  # [B, 4F]
        stat_out = self.stat_mlp(stats)                      # [B, H]
        stat_out = stat_out.unsqueeze(1).expand(-1, W, -1)   # [B, W, H]

        # --- (4) Combine paths ---
        weights = torch.softmax(self.alpha, dim=0)
        combined = (
            weights[0] * lstm_out +
            weights[1] * freq_out +
            weights[2] * stat_out
        )  # [B, W, H]

        # --- (5) Final projection ---
        out = self.fc_out(combined)  # [B, W, N]

        # --- (6) Flatten if N == 1 ---
        if self.output_dim == 1:
            out = out.squeeze(-1)  # → [B, W]

        return out


# ----------------------------------------------------
# PINN-Training Loop
# ----------------------------------------------------
def pinn_loss_lstm(model: nn.Module, X: torch.Tensor, T: torch.Tensor, P: torch.Tensor,
                   t: torch.Tensor, T0: torch.Tensor, dt: torch.Tensor, R: float, C: float, N_nodes,
                   Tamb: torch.Tensor, Tmin: float, Tmax: float, lambda_phys: float = 1.0, lambda_init: float = 1.0) \
                    -> tuple[torch.Tensor, float, float, float]:
    """
    Physics-informed loss for LSTM-PINN: data + physics + initial condition.
    """
    T_pred = model(X)
    dTdt_pred = gradient(T_pred, dt) * (Tmax - Tmin)

    T_t = T_pred * (Tmax - Tmin) + Tmin
    Tamb_phys = Tamb * (Tmax - Tmin) + Tmin

    if N_nodes == 1:
        rhs_total = (1.0 / C) * P - (1.0 / (R * C)) * (T_t - Tamb_phys)
        weights = torch.exp(-t / (R * C)).unsqueeze(0)
    else:
        rhs_total = 0 * dTdt_pred
        weights = torch.exp(-t / (np.min(R * C))).unsqueeze(0)
        for i in range(0, N_nodes):
            rhs = (1.0 / C[i]) * P - (1.0 / (R[i] * C[i])) * (T_t - Tamb_phys)
            rhs_total = rhs_total + rhs
    res = dTdt_pred - rhs_total

    ic_mse = torch.mean(weights * ((T_t - T0) / (Tmax - Tmin)) ** 2)
    data_mse = torch.mean((T_pred - T) ** 2)
    phys_mse = torch.mean(res ** 2)
    total = data_mse + lambda_phys * phys_mse + lambda_init * ic_mse

    return total, data_mse.item(), lambda_phys * phys_mse.item(), lambda_init * ic_mse.item()
