import torch
from geodesic_toolbox import CoMetric

VALID_OMEGA_TYPES = ["round", "constant", "random"]


class ConstantOmega(torch.nn.Module):
    def __init__(self, cometric: CoMetric, beta: float = 1.0):
        super().__init__()
        self.cometric = cometric
        self.beta = beta

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        omega = torch.zeros_like(z)
        omega[:, 0] = self.beta
        return omega


class RoundOmega(torch.nn.Module):
    def __init__(self, cometric: CoMetric, freq: float = 1.0):
        super().__init__()
        self.cometric = cometric
        self.freq = freq

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        omega = z.clone()
        omega[:, 0] = -self.freq * z[:, 1]
        omega[:, 1] = self.freq * z[:, 0]
        norm_omega = self.cometric.cometric(z, omega)
        omega = omega / (norm_omega.unsqueeze(1) + 1e-8)  # Avoid division by zero
        return omega


class RandomOmega(torch.nn.Module):
    def __init__(self, cometric: CoMetric, latent_dim: int = 2, hidden_dims: list = [64, 64]):
        super().__init__()
        self.cometric = cometric
        layers = []
        input_dim = latent_dim
        for hidden_dim in hidden_dims:
            layers.append(torch.nn.Linear(input_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            input_dim = hidden_dim
        layers.append(torch.nn.Linear(input_dim, 2))  # Output dimension is 2 for omega
        self.model = torch.nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        omega = self.model(z)
        norm_omega = self.cometric.cometric(z, omega)
        omega = omega / (norm_omega.unsqueeze(1) + 1e-8)  # Avoid division by zero
        return omega


def get_omega(omega_type: str, cometric: CoMetric, freq: float = 1.0, beta: float = 1.0):
    if omega_type == "round":
        return RoundOmega(cometric, freq=freq)
    elif omega_type == "constant":
        return ConstantOmega(cometric, beta=beta)
    elif omega_type == "random":
        return RandomOmega(cometric, latent_dim=2, hidden_dims=[64, 64])
    else:
        raise ValueError(
            f"Invalid omega_type '{omega_type}'. Valid options are {VALID_OMEGA_TYPES}."
        )
