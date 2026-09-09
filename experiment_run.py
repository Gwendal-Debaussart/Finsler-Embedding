import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from utils import *
from geodesic_toolbox import *

VALID_OMEGA_TYPES = ["round", "constant", "random"]


def parse_args():
    parser = ArgumentParser(description="Run the experiment for estimating b from data.")
    parser.add_argument("--N", type=int, default=5000, help="Number of data points.")
    parser.add_argument("--m", type=int, default=2, help="Dimension of the manifold.")
    parser.add_argument(
        "--beta", type=float, default=0.5, help="Beta parameter for Randers metric."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the computations on (e.g., 'cpu' or 'cuda').",
    )
    parser.add_argument(
        "--omega_type",
        type=str,
        default="round",
        choices=VALID_OMEGA_TYPES,
        help="Type of omega to use: 'round' or 'constant'.",
    )
    parser.add_argument(
        "--export_path",
        type=Path,
        default=Path("./results") / datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Path to export results.",
    )
    args = parser.parse_args()
    return args


@dataclass
class ExperimentConfig:
    N: int = 5000
    m: int = 2
    beta: float = 0.5
    device: str = "cpu"
    omega_type: str = "round"
    export_path: Path = Path("./results") / datetime.now().strftime("%Y%m%d_%H%M%S")


def plot_mf_and_omega(
    base_cometric: CoMetric, X: torch.Tensor, randers_metric: RandersMetrics, bounds: tuple
) -> tuple:
    mf = get_mf_image(base_cometric, X, bounds)
    omega_X = randers_metric.omega(X) * randers_metric.beta

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    im = axes[0].imshow(mf.cpu().numpy(), cmap="seismic", origin="lower", extent=bounds)
    t_cbar = axes[0].scatter(X[:, 0].cpu(), X[:, 1].cpu(), s=10, edgecolor="k", linewidth=0.5)
    axes[0].set_xlim(bounds[0], bounds[1])
    axes[0].set_ylim(bounds[2], bounds[3])
    axes[0].set_title("Density of the Dataset")
    axes[1].scatter(X[:, 0].cpu(), X[:, 1].cpu(), s=10)
    skip = 20  # Adjust this value to change the density of the quiver plot
    axes[1].quiver(
        X[::skip, 0].cpu(),
        X[::skip, 1].cpu(),
        omega_X[::skip, 0].cpu(),
        omega_X[::skip, 1].cpu(),
        color='red',
        scale=1,
        angles='xy',
        scale_units='xy',
    )
    axes[1].set_title("Dataset with Omega vector field")
    for ax in axes:
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_aspect('equal', adjustable='box')
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    fig.colorbar(t_cbar, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig, axes


def sample_uniform(
    n_samples: int, bounds: tuple = (-2, 2, -2, 2), device: str = "cpu"
) -> torch.Tensor:
    x = np.random.uniform(bounds[0], bounds[1], n_samples)
    y = np.random.uniform(bounds[2], bounds[3], n_samples)
    samples = np.stack([x, y], axis=1)
    return torch.from_numpy(samples).float().to(device)


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


class ConstantOmega(torch.nn.Module):
    def __init__(self, cometric: CoMetric, beta: float = 1.0):
        super().__init__()
        self.cometric = cometric
        self.beta = beta

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        omega = torch.zeros_like(z)
        omega[:, 0] = self.beta
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


def export_config(cfg: ExperimentConfig):
    config_dict = {
        "N": cfg.N,
        "m": cfg.m,
        "beta": cfg.beta,
        "device": cfg.device,
        "omega_type": cfg.omega_type,
        "export_path": str(cfg.export_path),
    }
    with open(cfg.export_path / "experiment_config.json", "w") as f:
        json.dump(config_dict, f, indent=4)


def main(cfg: ExperimentConfig):

    ######################################
    # Setup data and Randers metric
    ######################################
    X = sample_uniform(cfg.N, bounds=(-2, 2, -2, 2))
    X = X.to(torch.float).to(cfg.device)
    bounds = get_bounds(X, margin=0.01)

    base_cometric = IdentityCoMetric().to(cfg.device)
    omega_true = get_omega(cfg.omega_type, base_cometric).to(cfg.device)
    randers_metric = RandersMetrics(
        base_cometric=base_cometric,
        omega=omega_true,
        beta=cfg.beta,
    )

    edges = get_knn_graph(X, n_neighbors=10, device=cfg.device)
    print(f"Found {edges.shape[0]} edges in the KNN graph.")
    dst_edges = construct_distance_matrix(X, edges, randers_metric)

    fig, axes = plot_mf_and_omega(base_cometric, X, randers_metric, bounds)
    fig.savefig(cfg.export_path / "mf_and_omega.png", dpi=300)
    plt.close()

    ######################################
    # Setup the operators and compute the
    # values of f, Lf, and grad f
    ######################################
    epsilon = compute_epsilon_empirical(edges, dst_edges, cfg.N) * 0.5
    W, mu_0, mu_1 = gaussian_kernel(edges, dst_edges, epsilon, cfg.N, cfg.m)
    # W, mu_0, mu_1 = laplacian_kernel(edges, dst_edges, epsilon,cfg.N,cfg.m)
    c_km = -mu_1 / mu_0 * (cfg.m + 1) / cfg.m
    L_theta_s, L_theta_a = construct_operators(W, epsilon, theta=1)

    f_values_m, Lf_values_m, f_grad_values_m = mexican_family(
        X, L_theta_a, scale_list=[0.1, 0.2, 0.3, 0.4, 0.5]
    )
    f_values_g, Lf_values_g, f_grad_values_g = gaussian_family(
        X, L_theta_a, sigma_list=[0.1, 0.2, 0.3, 0.4, 0.5]
    )

    f_values = torch.cat([f_values_m, f_values_g], dim=0)
    Lf_values = torch.cat([Lf_values_m, Lf_values_g], dim=0)
    f_grad_values = torch.cat([f_grad_values_m, f_grad_values_g], dim=0)

    assert torch.all(torch.isfinite(f_grad_values)), "f_grad_values contains NaN or Inf"
    assert torch.all(torch.isfinite(Lf_values)), "Lf_values contains NaN or Inf"

    ######################################
    # Verify that the true values of b and
    # v are consistent with the computed Lf_values
    ######################################
    ## On a constant function
    f_values_constant = torch.ones((1, X.shape[0]), device=cfg.device)
    Lf_values_constant = L_theta_a @ f_values_constant.T  # (N, 1)
    Lf_values_constant = Lf_values_constant.T  # (1, N)
    f_grad_values_constant = torch.zeros(
        (1, X.shape[0], X.shape[1]), device=cfg.device
    )  # (1, N, D)

    print(f"Lf_values_constant should be zero")
    print(
        f"Lf_values_constant: mean = {Lf_values_constant.mean().item():.2e}, std = {Lf_values_constant.std().item():.2e}, min = {Lf_values_constant.min().item():.2e}, max = {Lf_values_constant.max().item():.2e}"
    )

    ax = sns.histplot(
        Lf_values_constant.flatten().cpu().detach().numpy(), bins=20, color='blue', alpha=0.7
    )
    ax.set_title('Distribution of $L[1](x_i)$ for constant function')
    ax.vlines(
        x=0,
        ymin=0,
        ymax=ax.get_ylim()[1],
        colors='red',
        linestyles='dashed',
        label='y=0',
    )
    fig = ax.get_figure()
    fig.savefig(cfg.export_path / "Lf_values_constant_distribution.png", dpi=300)
    plt.close()

    ## On the test functions
    b_true = randers_metric.omega(X) * randers_metric.beta
    v_true = b_to_v(b_true, base_cometric.cometric_tensor(X))
    v_true_norm = v_true.norm(dim=1)
    b_true_hat = v_to_b(v_true, base_cometric.metric_tensor(X))
    true_rhs = c_km * torch.einsum('n d, k n d -> k n', v_true, f_grad_values)
    print(
        f"Max error between b_true and b_true_hat: {(b_true_hat - b_true).norm(dim=1).max()}"
    )

    ######################################
    # Difference between true_rhs and Lf_values
    ######################################
    delta_rhs = true_rhs - Lf_values
    delta_rhs_norm = torch.norm(delta_rhs, dim=1)

    print(r'Delta_rhs_norm = $lVert Lf_k(x_i) - c_km <v(x_i), nabla f_k(x_i)>rVert$')
    print(
        f"Delta_rhs_norm: mean = {delta_rhs_norm.mean().item():.2f}, std = {delta_rhs_norm.std().item():.2f}, min = {delta_rhs_norm.min().item():.2f}, max = {delta_rhs_norm.max().item():.2f}"
    )
    sns.histplot(delta_rhs.flatten().cpu().detach().numpy(), bins=20, color='blue', alpha=0.7)
    plt.gca().set_yscale('log')
    plt.title(r'Distribution of $Lf_k(x_i) - c_km <v(x_i), nabla f_k(x_i)>$')
    plt.savefig(cfg.export_path / "delta_rhs_distribution.png", dpi=300)
    plt.close()

    ######################################
    # Ratio between true_rhs and Lf_values
    ######################################
    ratio_rhs = true_rhs / (Lf_values + 1e-8)  # Avoid division by zero
    ratio_rhs_mean = ratio_rhs.mean(dim=1)
    print(f"Ratio_rhs_mean should be 1")
    print(
        f"Ratio_rhs_mean: mean = {ratio_rhs_mean.mean().item():.2f}, std = {ratio_rhs_mean.std().item():.2f}, min = {ratio_rhs_mean.min().item():.2f}, max = {ratio_rhs_mean.max().item():.2f}"
    )
    sns.histplot(
        ratio_rhs_mean.flatten().cpu().detach().numpy(), bins=20, color='blue', alpha=0.7
    )
    plt.gca().set_yscale('log')
    plt.title(r'Distribution of $\frac{Lf_k(x_i)}{c_km <v(x_i), nabla f_k(x_i)>}$')
    plt.savefig(cfg.export_path / "ratio_rhs_distribution.png", dpi=300)
    plt.close()

    ######################################
    # Learn the vector field v using a
    # deep neural network
    ######################################
    v_model = V_estimator(hidden_dims=[64, 64], dim=2).to(cfg.device)
    omega_model = OmegaFromV(v_estimator=v_model, cometric=base_cometric).to(cfg.device)
    randers_model = RandersMetrics(
        base_cometric=base_cometric, omega=omega_model, beta=1.0
    ).to(cfg.device)
    loss_list = learn_v(
        X, f_values, Lf_values, f_grad_values, c_km, v_model, device=cfg.device
    )

    plt.plot(loss_list)
    plt.title("Training Loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(cfg.export_path / "training_loss.png", dpi=300)
    plt.close()

    v_hat_deep = v_model(X).detach()
    b_hat_deep = omega_model(X).detach()
    fig, axes = plot_side_by_side(X, b_true, b_hat_deep)
    plt.savefig(cfg.export_path / "b_true_vs_b_hat_deep.png", dpi=300)
    plt.close()

    cosim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    cosine_similarity_deep = cosim(v_hat_deep, v_true)
    plt.hist(cosine_similarity_deep.cpu().detach().numpy(), bins=50, color='blue', alpha=0.7)
    plt.title("Cosine Similarity between True and Estimated v (Deep Model)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.savefig(cfg.export_path / "cosine_similarity_deep.png", dpi=300)
    plt.close()

    v_hat_deep_norm = v_hat_deep.norm(dim=1) ** 2
    alpha_ratio = torch.einsum("nd,nd->n", v_hat_deep, v_true) / (
        v_true_norm + 1e-8
    )  # Avoid division by zero
    print(f"Alpha ratio should be 1")
    print(
        f"Alpha ratio: mean = {alpha_ratio.mean().item():.2f}, std = {alpha_ratio.std().item():.2f}, min = {alpha_ratio.min().item():.2f}, max = {alpha_ratio.max().item():.2f}"
    )
    plt.hist(alpha_ratio.cpu().detach().numpy(), bins=50, color='blue', alpha=0.7)
    plt.xlabel('Alpha Ratio')
    plt.ylabel('Frequency')
    plt.title(
        r'Distribution of $\alpha = \frac{<v_{hat}(x_i), v_{true}(x_i)>}{lVert v_{true}(x_i)rVert^2}$'
    )
    plt.savefig(cfg.export_path / "alpha_ratio_distribution.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    cfg = ExperimentConfig(
        N=args.N,
        m=args.m,
        beta=args.beta,
        device=args.device,
        omega_type=args.omega_type,
        export_path=args.export_path,
    )
    cfg.export_path.mkdir(parents=True, exist_ok=True)
    export_config(cfg)
    main(cfg)
