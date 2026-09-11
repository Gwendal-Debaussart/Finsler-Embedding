import torch
from finsler_embedding.experiment_run import ExperimentConfig

VALID_TEST_FUNCTION_TYPES = [
    "mexican",
    "gaussian",
    "mexican_and_gaussian",
    "coordinates",
]

def bump_function(x: torch.Tensor, center: torch.Tensor, radius: float) -> torch.Tensor:
    """
    A smooth bump function that is 1 at the center and smoothly decays to 0 at the radius.

    Parameters:
    ----------
    x : torch.Tensor (B, D)
        The input points in D-dimensional space.
    center : torch.Tensor (D,)
        The center of the bump function.
    radius : float
        The radius of the bump function.

    Returns:
    -------
    torch.Tensor (B,)
        The values of the bump function at the input points.
    """
    distance = torch.norm(x - center, dim=-1)
    return torch.where(
        distance < radius,
        torch.exp(-1 / (1 - (distance / radius) ** 2)),
        torch.zeros_like(distance),
    )


def gaussian(x: torch.Tensor, center: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    A Gaussian function centered at the given point.

    Parameters:
    ----------
    x : torch.Tensor (B, D)
        The input points in D-dimensional space.
    center : torch.Tensor (D,)
        The center of the Gaussian function.
    sigma : float
        The standard deviation of the Gaussian function.

    Returns:
    -------
    torch.Tensor (B,)
        The values of the Gaussian function at the input points.
    """
    distance_squared = torch.sum((x - center) ** 2, dim=-1)
    return torch.exp(-distance_squared / (2 * sigma**2))


def random_sin(X: torch.Tensor) -> torch.Tensor:
    """
    A random sine function with random frequency and phase.

    Parameters:
    ----------
    X : torch.Tensor (B, D)
        The input points in D-dimensional space.

    Returns:
    -------
    torch.Tensor (B,)
        The values of the random sine function at the input points.
    """
    phase = torch.rand(1) * 2 * torch.pi
    frequency = torch.rand(1) * 5 + 1  # Random frequency between 1 and 6
    return torch.sin(frequency * X + phase).sum(dim=-1)


def mexican_hat(X):
    """
    A Mexican hat function (Ricker wavelet) in D dimensions.

    Parameters:
    ----------
    X : torch.Tensor (B, D)
        The input points in D-dimensional space.

    Returns:
    -------
    torch.Tensor (B,)
        The values of the Mexican hat function at the input points.
    """
    norm_squared = torch.sum(X**2, dim=-1)
    return (1 - norm_squared) * torch.exp(-norm_squared / 2)


def scaled_mexican_hat(X, scale: float, center: torch.Tensor) -> torch.Tensor:
    """
    A scaled Mexican hat function (Ricker wavelet) in D dimensions.

    Parameters:
    ----------
    X : torch.Tensor (B, D)
        The input points in D-dimensional space.
    scale : float
        The scale parameter of the Mexican hat function.
    center : torch.Tensor (D,)
        The center parameter of the Mexican hat function.

    Returns:
    -------
    torch.Tensor (B,)
        The values of the scaled Mexican hat function at the input points.
    """
    return mexican_hat((X - center) / scale) / scale ** (X.shape[1] / 2)


def gaussian_family(
    X: torch.Tensor,
    L_theta_a: torch.Tensor,
    sigma_list: list[float],
    K: int,
):
    """
    A family of Gaussian functions with different scales.

    Parameters:
    ----------
    X : torch.Tensor (B, D)
        The input points in D-dimensional space.
    L_theta_a : torch.Tensor (N, N)
        The antisymmetric part of the normalized Laplacian operator.
    sigma_list : list[float]
        A list of standard deviations for the Gaussian functions.
    K : int
        The total number of test functions generated. Thus each sigma will have K / len(sigma_list) test functions.

    Returns:
    -------
    f_values : torch.Tensor (K, N)
        The values of the Gaussian functions at the input points.
    Lf_values : torch.Tensor (K, N)
        The values of the Laplacian applied to the Gaussian functions at the input points.
    f_grad_values : torch.Tensor (K, N, D)
        The gradients of the Gaussian functions at the input points.
    """
    K_ = K // len(sigma_list)  # Number of test functions per sigma

    f_values = []
    Lf_values = []
    f_grad_values = []

    for sigma in sigma_list:
        f_values_radius = torch.zeros((K_, X.shape[0]))
        Lf_values_radius = torch.zeros((K_, X.shape[0]))
        f_grad_values_radius = torch.zeros((K_, X.shape[0], X.shape[1]))
        X.requires_grad_()
        idx_i = torch.randint(0, X.shape[0], (K_,), device=X.device)
        for i in range(K_):
            center = X[idx_i[i]].detach()  # (D,)
            f_ = gaussian(X, center=center.detach(), sigma=sigma)  # (N,)
            f_grad_ = torch.autograd.grad(f_.sum(), X, create_graph=True)[0]  # (N, D)

            f_values_radius[i] = f_
            f_grad_values_radius[i] = f_grad_
            Lf_values_radius[i] = L_theta_a @ f_  # (N,)

        f_values.append(f_values_radius.detach())
        Lf_values.append(Lf_values_radius.detach())
        f_grad_values.append(f_grad_values_radius.detach())

    f_values = torch.cat(f_values, dim=0)  # (K, N)
    Lf_values: torch.Tensor = torch.cat(Lf_values, dim=0)  # (K, N)
    f_grad_values = torch.cat(f_grad_values, dim=0)  # (K, N, D)

    return f_values, Lf_values, f_grad_values


def mexican_family(X: torch.Tensor, L_theta_a: torch.Tensor, scale_list: list[float], K: int):
    """
    A family of Mexican hat functions with different scales.

    Parameters:
    ----------
    X : torch.Tensor (B, D)
        The input points in D-dimensional space.
    L_theta_a : torch.Tensor (N, N)
        The antisymmetric part of the normalized Laplacian operator.
    scale_list : list[float]
        A list of scales for the Mexican hat functions.

    Returns:
    -------
    f_values : torch.Tensor (K, N)
        The values of the Mexican hat functions at the input points.
    Lf_values : torch.Tensor (K, N)
        The values of the Laplacian applied to the Mexican hat functions at the input points.
    f_grad_values : torch.Tensor (K, N, D)
        The gradients of the Mexican hat functions at the input points.
    K : int
        The total number of test functions generated. Thus each scale will have K / len(scale_list) test functions.
    """
    K_ = K // len(scale_list)  # Number of test functions per scale

    f_values = []
    Lf_values = []
    f_grad_values = []

    for scale in scale_list:
        f_values_radius = torch.zeros((K_, X.shape[0]))
        Lf_values_radius = torch.zeros((K_, X.shape[0]))
        f_grad_values_radius = torch.zeros((K_, X.shape[0], X.shape[1]))
        X.requires_grad_()
        idx_i = torch.randint(0, X.shape[0], (K_,), device=X.device)
        for i in range(K_):
            center = X[idx_i[i]].detach()  # (D,)
            f_ = scaled_mexican_hat(X, center=center.detach(), scale=scale)  # (N,)
            f_grad_ = torch.autograd.grad(f_.sum(), X, create_graph=True)[0]  # (N, D)

            f_values_radius[i] = f_
            f_grad_values_radius[i] = f_grad_
            Lf_values_radius[i] = L_theta_a @ f_  # (N,)

        f_values.append(f_values_radius.detach())
        Lf_values.append(Lf_values_radius.detach())
        f_grad_values.append(f_grad_values_radius.detach())

    f_values = torch.cat(f_values, dim=0)  # (K, N)
    Lf_values: torch.Tensor = torch.cat(Lf_values, dim=0)  # (K, N)
    f_grad_values = torch.cat(f_grad_values, dim=0)  # (K, N, D)

    return f_values, Lf_values, f_grad_values


def get_mexian_tf(cfg: ExperimentConfig, X: torch.Tensor, L_theta_a: torch.Tensor):
    f_values_m, Lf_values_m, f_grad_values_m = mexican_family(
        X,
        L_theta_a,
        scale_list=[0.1, 0.2, 0.3, 0.4, 0.5],
        K=cfg.K,
    )
    return f_values_m, Lf_values_m, f_grad_values_m


def get_gaussian_tf(cfg: ExperimentConfig, X: torch.Tensor, L_theta_a: torch.Tensor):
    f_values_g, Lf_values_g, f_grad_values_g = gaussian_family(
        X,
        L_theta_a,
        sigma_list=[0.1, 0.2, 0.3, 0.4, 0.5],
        K=cfg.K,
    )
    return f_values_g, Lf_values_g, f_grad_values_g


def mexican_and_gaussian_tf(cfg: ExperimentConfig, X: torch.Tensor, L_theta_a: torch.Tensor):
    f_values_m, Lf_values_m, f_grad_values_m = mexican_family(
        X,
        L_theta_a,
        scale_list=[0.1, 0.2, 0.3, 0.4, 0.5],
        K=cfg.K // 2,
    )
    f_values_g, Lf_values_g, f_grad_values_g = gaussian_family(
        X,
        L_theta_a,
        sigma_list=[0.1, 0.2, 0.3, 0.4, 0.5],
        K=cfg.K // 2,
    )

    f_values = torch.cat([f_values_m, f_values_g], dim=0)
    Lf_values = torch.cat([Lf_values_m, Lf_values_g], dim=0)
    f_grad_values = torch.cat([f_grad_values_m, f_grad_values_g], dim=0)

    return f_values, Lf_values, f_grad_values


def coordinates_tf(
    cfg: ExperimentConfig, X: torch.Tensor, L_theta_a: torch.Tensor, coord: int = 0
):
    f_values = X[:, coord].unsqueeze(0)  # (1, N)
    Lf_values = L_theta_a @ f_values.T  # (N, 1)
    Lf_values = Lf_values.T  # (1, N)
    f_grad_values = torch.zeros((1, X.shape[0], X.shape[1]), device=cfg.device)  # (1, N, D)
    f_grad_values[0, :, coord] = 1.0

    return f_values, Lf_values, f_grad_values


def prepare_test_functions(cfg: ExperimentConfig, X: torch.Tensor, L_theta_a: torch.Tensor):
    if cfg.test_function_type == "mexican":
        f_values, Lf_values, f_grad_values = get_mexian_tf(cfg, X, L_theta_a)
    elif cfg.test_function_type == "gaussian":
        f_values, Lf_values, f_grad_values = get_gaussian_tf(cfg, X, L_theta_a)
    elif cfg.test_function_type == "mexican_and_gaussian":
        f_values, Lf_values, f_grad_values = mexican_and_gaussian_tf(cfg, X, L_theta_a)
    elif cfg.test_function_type == "coordinates":
        f_values, Lf_values, f_grad_values = coordinates_tf(cfg, X, L_theta_a)
    else:
        raise ValueError(f"Unknown test function type: {cfg.test_function_type}")

    assert torch.all(torch.isfinite(f_grad_values)), "f_grad_values contains NaN or Inf"
    assert torch.all(torch.isfinite(Lf_values)), "Lf_values contains NaN or Inf"
    return f_values, Lf_values, f_grad_values
