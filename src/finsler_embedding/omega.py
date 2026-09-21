import torch
from geodesic_toolbox import CoMetric

VALID_OMEGA_TYPES = ["round", "constant", "random", "swiss_roll"]


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
        layers.append(torch.nn.Linear(input_dim, latent_dim))
        self.model = torch.nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        omega = self.model(z)
        norm_omega = self.cometric.cometric(z, omega)
        omega = omega / (norm_omega.unsqueeze(1) + 1e-8)  # Avoid division by zero
        return omega


class OmegaSwissRoll(torch.nn.Module):
    """Returns a vector tangent to the swiss roll manifold at a given point in 3D space."""

    def __init__(self, cometric: CoMetric, jitter: float = 1e-2):
        super().__init__()
        self.cometric = cometric
        self.jitter = jitter

    def forward(self, x):
        """
        Computes the tangent vector to the swiss roll manifold at the given 3D point x.

        Parameters:
        x (torch.Tensor): A tensor of shape (N, 3) representing N points in 3D space.

        Returns:
        torch.Tensor: A tensor of shape (N, 3) representing the tangent vectors at the given points.
        """
        # Compute the angle theta based on the x and z coordinates
        theta = torch.atan2(x[:, 2], x[:, 0])

        # Compute the tangent vector components
        tangent_x = -torch.sin(theta)
        tangent_y = torch.zeros_like(tangent_x)  # No change in y-direction
        tangent_z = torch.cos(theta)

        # Stack the components to form the tangent vector
        tangent_vector = torch.stack((tangent_x, tangent_y, tangent_z), dim=1)
        tangent_vector += self.jitter * torch.randn_like(
            tangent_vector
        )  # Add jitter for numerical stability

        omega_norm = self.cometric.cometric(x, tangent_vector)
        omega = (
            tangent_vector / omega_norm[:, None]
        )  # Normalize the tangent vector using the cometric

        return omega


class OmegaSwissRollNormal(torch.nn.Module):
    """Returns a vector normal to the swiss roll manifold at a given point in 3D space."""

    def __init__(self, cometric: CoMetric, jitter: float = 1e-2):
        super().__init__()
        self.cometric = cometric
        self.jitter = jitter

    def forward(self, x):
        """
        Computes the normal vector to the swiss roll manifold at the given 3D point x.

        Parameters:
        x (torch.Tensor): A tensor of shape (N, 3) representing N points in 3D space.

        Returns:
        torch.Tensor: A tensor of shape (N, 3) representing the normal vectors at the given points.
        """
        # Compute the angle theta based on the x and z coordinates
        theta = torch.atan2(x[:, 2], x[:, 0])

        # Compute the normal vector components
        normal_x = torch.cos(theta)
        normal_y = torch.zeros_like(normal_x)  # No change in y-direction
        normal_z = torch.sin(theta)

        # Stack the components to form the normal vector
        normal_vector = torch.stack((normal_x, normal_y, normal_z), dim=1)
        normal_vector += self.jitter * torch.randn_like(
            normal_vector
        )  # Add jitter for numerical stability

        omega_norm = self.cometric.cometric(x, normal_vector)
        omega = (
            normal_vector / omega_norm[:, None]
        )  # Normalize the normal vector using the cometric

        return omega


class CircularSphereOmega(torch.nn.Module):
    """
    Returns a vector tangent to the circular sphere manifold at a given point in 3D space.
    """

    def __init__(self, cometric: CoMetric, jitter: float = 1e-2):
        super().__init__()
        self.cometric = cometric
        self.jitter = jitter

    def forward(self, x):
        """
        Computes the tangent vector to the circular sphere manifold at the given 3D point x.

        Parameters:
        x (torch.Tensor): A tensor of shape (N, 3) representing N points in 3D space.

        Returns:
        torch.Tensor: A tensor of shape (N, 3) representing the tangent vectors at the given points.
        """
        # Compute the tangent vector components
        tangent_x = -x[:, 1]
        tangent_y = x[:, 0]
        tangent_z = torch.zeros_like(tangent_x)  # No change in z-direction

        # Stack the components to form the tangent vector
        tangent_vector = torch.stack((tangent_x, tangent_y, tangent_z), dim=1)
        tangent_vector += self.jitter * torch.randn_like(
            tangent_vector
        )  # Add jitter for numerical stability

        omega_norm = self.cometric.cometric(x, tangent_vector)
        omega = (
            tangent_vector / omega_norm[:, None]
        )  # Normalize the tangent vector using the cometric

        return omega



def spiral_omega(z: torch.Tensor, freq: float) -> torch.Tensor:
    """
    Compute the omega vector field for a spiral.

    Parameters:
    ----------
    z : torch.Tensor (n_points, 2)
        The input points in 2D space.
    freq : float
        The frequency of the spiral.

    Returns:
    -------
    torch.Tensor (n_points, 2)
        The omega vector field at the input points.
    """
    x = z[:, 0]
    y = z[:, 1]
    t = torch.sqrt(x**2 + y**2)
    omega_x = -freq * y / (t + 1e-8)  # Avoid division by zero
    omega_y = freq * x / (t + 1e-8)   # Avoid division by zero
    omega = torch.stack([omega_x, omega_y], dim=1)
    return omega / torch.norm(omega, dim=1, keepdim=True)  # Normalize the vector field

class OmegaSpiral(torch.nn.Module):
    def __init__(self, cometric:CoMetric, freq: float = 1.0):
        super().__init__()
        self.cometric = cometric
        self.freq = freq

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        omega = spiral_omega(z, freq=self.freq)
        norm_omega = self.cometric.cometric(z, omega)
        omega = omega / (norm_omega.unsqueeze(1) + 1e-8)  # Avoid division by zero
        return omega

def get_omega(
    omega_type: str, cometric: CoMetric, freq: float = 1.0, beta: float = 1.0, dim=2
):
    if omega_type == "round":
        return RoundOmega(cometric, freq=freq)
    elif omega_type == "constant":
        return ConstantOmega(cometric, beta=beta)
    elif omega_type == "random":
        return RandomOmega(cometric, latent_dim=dim, hidden_dims=[64, 64])
    elif omega_type == "swiss_roll":
        return OmegaSwissRoll(cometric)
    elif omega_type == "swiss_roll_normal":
        return OmegaSwissRollNormal(cometric)
    elif omega_type == "circular_sphere":
        return CircularSphereOmega(cometric)
    elif omega_type == "spiral":
        return OmegaSpiral(cometric, freq=freq)
    else:
        raise ValueError(
            f"Invalid omega_type '{omega_type}'. Valid options are {VALID_OMEGA_TYPES}."
        )
