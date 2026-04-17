import torch
import torch.nn as nn

class CardiovascularPINN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(CardiovascularPINN, self).__init__()
        # Standard Feedforward Network
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)  # Predicting e.g., sysBP and diaBP
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def physics_loss(self, sys_bp: torch.Tensor, dia_bp: torch.Tensor) -> torch.Tensor:
        """
        Enforces physiological constraints.
        E.g., Mean Arterial Pressure (MAP) approximation: MAP ≈ diaBP + (sysBP - diaBP)/3
        Pulse Pressure (PP) must be > 0 (sysBP > diaBP)
        """
        # Constraint 1: Systolic BP must be strictly greater than Diastolic BP
        pulse_pressure = sys_bp - dia_bp
        pp_violation = torch.relu(-pulse_pressure) # Penalize if diaBP >= sysBP
        
        # Constraint 2: MAP should generally fall within physiological survival limits (e.g., 60-110 mmHg)
        map_approx = dia_bp + (pulse_pressure / 3.0)
        map_low_violation = torch.relu(60.0 - map_approx)
        map_high_violation = torch.relu(map_approx - 110.0)
        
        return torch.mean(pp_violation + map_low_violation + map_high_violation)

    def compute_total_loss(self, predictions: torch.Tensor, targets: torch.Tensor, lambda_phys: float = 0.5) -> torch.Tensor:
        """
        Combines standard MSE loss with the Physics-Informed loss.
        """
        mse_loss = torch.nn.functional.mse_loss(predictions, targets)
        
        sys_bp_pred = predictions[:, 0]
        dia_bp_pred = predictions[:, 1]
        phys_loss = self.physics_loss(sys_bp_pred, dia_bp_pred)
        
        return mse_loss + (lambda_phys * phys_loss)
