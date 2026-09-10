import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from geodesic_toolbox import *
from finsler_embedding.utils import *
import logging
from finsler_embedding.logger import setup_logger, update_log_file

LOGGER = setup_logger(name=None, log_file=None, level=logging.INFO)

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
        default=Path("/home/tblanchard/phd/Finsler-Embedding/results")
        / datetime.now().strftime("%Y%m%d_%H%M%S"),
        help="Path to export results.",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="If set, do not generate plots.",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=500,
        help="Number of test functions (Mexican Hat and Gaussian families).",
    )
    parser.add_argument(
        "--n_neighbors",
        type=int,
        default=10,
        help="Number of neighbors for KNN graph. Put -1 to use all points as neighbors (i.e., fully connected graph).",
    )
    args = parser.parse_args()
    return args


@dataclass
class ExperimentConfig:
    N: int = 5000
    m: int = 2
    beta: float = 0.5
    device: str = "cpu"
    omega_type: str = "constant"
    export_path: Path = Path("./results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    no_plot: bool = False
    K: int = 500  # Number of test functions (Mexican Hat and Gaussian families)
    n_neighbors: int = (
        10  # Number of neighbors for KNN graph. Put -1 to use all points as neighbors (i.e., fully connected graph)
    )


def sample_uniform(
    n_samples: int, bounds: tuple = (-2, 2, -2, 2), device: str = "cpu"
) -> torch.Tensor:
    x = np.random.uniform(bounds[0], bounds[1], n_samples)
    y = np.random.uniform(bounds[2], bounds[3], n_samples)
    samples = np.stack([x, y], axis=1)
    return torch.from_numpy(samples).float().to(device)


def sample_grid(
    n_samples: int, bounds: tuple = (-2, 2, -2, 2), device: str = "cpu"
) -> torch.Tensor:
    grid_size = int(np.sqrt(n_samples))
    x = np.linspace(bounds[0], bounds[1], grid_size)
    y = np.linspace(bounds[2], bounds[3], grid_size)
    xv, yv = np.meshgrid(x, y)
    samples = np.stack([xv.flatten(), yv.flatten()], axis=1)
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
    LOGGER.info(
        f"Experiment configuration exported to {cfg.export_path / 'experiment_config.json'}"
    )


def export_results(cfg: ExperimentConfig, results: dict[str, dict]):
    """
    Export the results of the experiment to a JSON file.

    Parameters:
    -----------
    cfg : ExperimentConfig
        The configuration of the experiment.
    results : dict
        A dictionary containing the results of the experiment. results["filename"] = {"mean": ..., "std": ..., "min": ..., "max": ...}
    """
    for key, value in results.items():
        if not isinstance(value, dict):
            raise ValueError(f"Value for key '{key}' must be a dictionary.")
        with open(cfg.export_path / f"{key}.json", "w") as f:
            json.dump(value, f, indent=4)


def train_vector_field_models(
    cfg: ExperimentConfig,
    X: torch.Tensor,
    base_cometric: CoMetric,
    c_km: torch.Tensor,
    f_values: torch.Tensor,
    Lf_values: torch.Tensor,
    f_grad_values: torch.Tensor,
) -> tuple[torch.nn.Module, torch.nn.Module, list[float]]:
    v_model = V_estimator(hidden_dims=[64, 64], dim=2).to(cfg.device)
    omega_model = OmegaFromV(v_estimator=v_model, cometric=base_cometric).to(cfg.device)
    loss_list = learn_v(
        X, f_values, Lf_values, f_grad_values, c_km, v_model, device=cfg.device
    )
    return v_model, omega_model, loss_list


def qqt_summary_statistics(values: torch.Tensor) -> dict[str, float]:
    res = {
        "mean": values.mean().item(),
        "std": values.std().item(),
        "min": values.min().item(),
        "max": values.max().item(),
        "quantile_5": values.quantile(0.05).item(),
        "quantile_25": values.quantile(0.25).item(),
        "median": values.quantile(0.50).item(),
        "quantile_75": values.quantile(0.75).item(),
        "quantile_95": values.quantile(0.95).item(),
    }
    return res


def main(cfg: ExperimentConfig):

    ######################################
    # Setup data and Randers metric
    ######################################
    X = sample_uniform(cfg.N, bounds=(-2, 2, -2, 2))
    # X = sample_grid(cfg.N, bounds=(-2, 2, -2, 2))
    # Shuffle the data to avoid any ordering effects
    X = X[torch.randperm(X.shape[0])]
    cfg.N = X.shape[0]  # Update N in case of grid sampling
    X = X.to(torch.float).to(cfg.device)
    bounds = get_bounds(X, margin=0.01)

    base_cometric = IdentityCoMetric().to(cfg.device)
    omega_true = get_omega(cfg.omega_type, base_cometric).to(cfg.device)
    randers_metric = RandersMetrics(
        base_cometric=base_cometric,
        omega=omega_true,
        beta=cfg.beta,
    )

    edges = get_knn_graph(X, n_neighbors=cfg.n_neighbors, device=cfg.device)
    LOGGER.info(f"Found {edges.shape[0]} edges in the KNN graph.")
    dst_edges = construct_distance_matrix(X, edges, randers_metric, use_approx=True)

    ######################################
    # Setup the operators and compute the
    # values of f, Lf, and grad f
    ######################################
    LOGGER.info("Computing epsilon, W, mu_0, mu_1, and operators L_theta_s and L_theta_a...")
    epsilon = compute_epsilon_empirical(X) * 10
    W, mu_0, mu_1 = gaussian_kernel(edges, dst_edges, epsilon, cfg.N, cfg.m)
    # W, mu_0, mu_1 = laplacian_kernel(edges, dst_edges, epsilon, cfg.N, cfg.m)
    c_km = -mu_1 / mu_0 * (cfg.m + 1) / cfg.m
    L_theta_s, L_theta_a = construct_operators(W, epsilon, theta=1)

    LOGGER.info(
        "Computing the values of f, Lf, and grad f for the Mexican Hat and Gaussian families..."
    )
    f_values_m, Lf_values_m, f_grad_values_m = mexican_family(
        X, L_theta_a, scale_list=[0.1, 0.2, 0.3, 0.4, 0.5], K=cfg.K // 2
    )
    f_values_g, Lf_values_g, f_grad_values_g = gaussian_family(
        X, L_theta_a, sigma_list=[0.1, 0.2, 0.3, 0.4, 0.5], K=cfg.K // 2
    )

    f_values = torch.cat([f_values_m, f_values_g], dim=0)
    Lf_values = torch.cat([Lf_values_m, Lf_values_g], dim=0)
    f_grad_values = torch.cat([f_grad_values_m, f_grad_values_g], dim=0)

    # Other easier test function
    # f_values = X[:, 0].unsqueeze(0)  # (1, N)
    # Lf_values = L_theta_a @ f_values.T  # (N, 1)
    # Lf_values = Lf_values.T  # (1, N)
    # f_grad_values = torch.zeros((1, X.shape[0], X.shape[1]), device=cfg.device)  # (1, N, D)
    # f_grad_values[0, :, 0] = 1.0

    assert torch.all(torch.isfinite(f_grad_values)), "f_grad_values contains NaN or Inf"
    assert torch.all(torch.isfinite(Lf_values)), "Lf_values contains NaN or Inf"
    LOGGER.info(f"Computed K = {f_values.shape[0]} test functions and their gradients.")

    ######################################
    # Learn the vector field v using a
    # deep neural network
    ######################################
    LOGGER.info("Training vector field models...")
    v_model, omega_model, loss_list = train_vector_field_models(
        cfg, X, base_cometric, c_km, f_values, Lf_values, f_grad_values
    )
    v_hat_deep = v_model(X).detach()
    b_hat_deep = omega_model(X).detach()

    #####################################
    # Compute some quantities of interest
    #####################################
    f_values_constant = torch.ones((1, X.shape[0]), device=cfg.device)
    Lf_values_constant = L_theta_a @ f_values_constant.T  # (N, 1)
    Lf_values_constant = Lf_values_constant.T  # (1, N)
    f_grad_values_constant = torch.zeros(
        (1, X.shape[0], X.shape[1]), device=cfg.device
    )  # (1, N, D)

    res_LF_value_constant = qqt_summary_statistics(Lf_values_constant.flatten())
    LOGGER.info(f"Lf_values_constant should be zero")
    LOGGER.info(
        f"Lf_values_constant: mean = {res_LF_value_constant['mean']:.2e}, std = {res_LF_value_constant['std']:.2e}, min = {res_LF_value_constant['min']:.2e}, max = {res_LF_value_constant['max']:.2e}"
    )

    b_true = randers_metric.omega(X) * randers_metric.beta
    v_true = b_to_v(b_true, base_cometric.cometric_tensor(X))
    v_true_norm = v_true.norm(dim=1)
    true_rhs = c_km * torch.einsum('n d, k n d -> k n', v_true, f_grad_values)

    delta_rhs = true_rhs - Lf_values
    delta_rhs_norm = torch.norm(delta_rhs, dim=1)

    res_delta_rhs_norm = qqt_summary_statistics(delta_rhs_norm)
    LOGGER.info(r'Delta_rhs_norm = $lVert Lf_k(x_i) - c_km <v(x_i), nabla f_k(x_i)>rVert$')
    LOGGER.info(
        f"Delta_rhs_norm: mean = {res_delta_rhs_norm['mean']:.2e}, std = {res_delta_rhs_norm['std']:.2e}, min = {res_delta_rhs_norm['min']:.2e}, max = {res_delta_rhs_norm['max']:.2e}"
    )

    ratio_rhs = true_rhs / (Lf_values + 1e-8)  # Avoid division by zero
    ratio_rhs_mean = ratio_rhs.mean(dim=1)
    res_ratio_rhs_mean = qqt_summary_statistics(ratio_rhs_mean)
    LOGGER.info(f"Ratio_rhs_mean should be 1")
    LOGGER.info(
        f"Ratio_rhs_mean: mean = {res_ratio_rhs_mean['mean']:.2e}, std = {res_ratio_rhs_mean['std']:.2e}, min = {res_ratio_rhs_mean['min']:.2e}, max = {res_ratio_rhs_mean['max']:.2e}"
    )

    cosim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    cosine_similarity_deep = cosim(b_hat_deep, b_true)
    res_cosine_similarity_deep = qqt_summary_statistics(cosine_similarity_deep)
    LOGGER.info(f"Cosine similarity between true b and estimated b (Deep Model) should be 1")
    LOGGER.info(
        f"Cosine similarity: mean = {res_cosine_similarity_deep['mean']:.2e}, std = {res_cosine_similarity_deep['std']:.2e}, min = {res_cosine_similarity_deep['min']:.2e}, max = {res_cosine_similarity_deep['max']:.2e}"
    )

    # We regress cst such that v_hat_deep = cst * v_true. We should have cst = 1 if the estimation is perfect.
    alpha_ratio = torch.einsum('n d, n d -> n', v_hat_deep, v_true) / (
        v_true_norm**2 + 1e-8
    )  # Avoid division by zero
    res_alpha_ratio = qqt_summary_statistics(alpha_ratio)
    LOGGER.info(f"Alpha ratio should be 1")
    LOGGER.info(
        f"Alpha ratio: mean = {res_alpha_ratio['mean']:.2e}, std = {res_alpha_ratio['std']:.2e}, min = {res_alpha_ratio['min']:.2e}, max = {res_alpha_ratio['max']:.2e}"
    )

    all_results = {
        "res_LF_value_constant": res_LF_value_constant,
        "res_delta_rhs_norm": res_delta_rhs_norm,
        "res_ratio_rhs_mean": res_ratio_rhs_mean,
        "res_cosine_similarity_deep": res_cosine_similarity_deep,
        "res_alpha_ratio": res_alpha_ratio,
    }
    export_results(cfg, all_results)

    ######################################
    # Plot the training loss and other distributions
    ######################################
    if cfg.no_plot:
        LOGGER.info("Plotting is disabled. Skipping plot generation.")
        return

    #######################################
    # Plot the density of the dataset and the
    # vector field omega
    #######################################
    LOGGER.info("Plotting the density of the dataset and the vector field omega...")
    fig, axes = plot_mf_and_omega(base_cometric, X, randers_metric, bounds)
    fig.savefig(cfg.export_path / "mf_and_omega.png", dpi=300)
    plt.close()

    LOGGER.info("Plotting the training loss...")
    loss_list = np.array(loss_list)
    plt.plot(loss_list)
    plt.title("Training Loss")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(cfg.export_path / "training_loss.png", dpi=300)
    plt.close()

    LOGGER.info("Plotting the comparison between true b and estimated b (Deep Model)...")
    fig, axes = plot_side_by_side(X, b_true, b_hat_deep)
    plt.savefig(cfg.export_path / "b_true_vs_b_hat_deep.png", dpi=300)
    plt.close()

    LOGGER.info("Plotting the distribution of Lf_values_constant...")
    ax = sns.histplot(
        Lf_values_constant.flatten().cpu().detach().numpy(), bins=20, color='blue', alpha=0.7
    )
    ax.set_title('Distribution of $L[1](x_i)$ ')
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

    LOGGER.info("Plotting the distribution of delta_rhs...")
    sns.histplot(delta_rhs.flatten().cpu().detach().numpy(), bins=20, color='blue', alpha=0.7)
    plt.gca().set_yscale('log')
    plt.title(r'Distribution of $Lf_k(x_i) - c_km <v(x_i), \nabla f_k(x_i)>$')
    plt.savefig(cfg.export_path / "delta_rhs_distribution.png", dpi=300)
    plt.close()

    LOGGER.info("Plotting the distribution of ratio_rhs_mean...")
    sns.histplot(
        ratio_rhs_mean.flatten().cpu().detach().numpy(), bins=20, color='blue', alpha=0.7
    )
    plt.gca().set_yscale('log')
    plt.title(r'Distribution of $\frac{Lf_k(x_i)}{c_km <v(x_i), \nabla f_k(x_i)>}$')
    plt.savefig(cfg.export_path / "ratio_rhs_distribution.png", dpi=300)
    plt.close()

    LOGGER.info(
        "Plotting the distribution of cosine similarity between true b and estimated b (Deep Model)..."
    )
    plt.hist(cosine_similarity_deep.cpu().detach().numpy(), bins=50, color='blue', alpha=0.7)
    plt.title("Cosine Similarity between True and Estimated b (Deep Model)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.savefig(cfg.export_path / "cosine_similarity_deep.png", dpi=300)
    plt.close()

    LOGGER.info("Plotting the distribution of alpha ratio...")
    plt.hist(alpha_ratio.cpu().detach().numpy(), bins=50, color='blue', alpha=0.7)
    plt.xlabel('Alpha Ratio')
    plt.ylabel('Frequency')
    plt.title(
        r'Distribution of $\alpha = \frac{<v_{hat}(x_i), v_{true}(x_i)>}{\|v_{true}(x_i)\|^2}$'
    )
    plt.savefig(cfg.export_path / "alpha_ratio_distribution.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    start_time = datetime.now()
    args = parse_args()
    cfg = ExperimentConfig(
        N=args.N,
        m=args.m,
        beta=args.beta,
        device=args.device,
        omega_type=args.omega_type,
        export_path=args.export_path,
        no_plot=args.no_plot,
        K=args.K,
        n_neighbors=args.n_neighbors,
    )
    cfg.export_path.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Results will be exported to: {cfg.export_path}")
    LOGGER.info(f"Experiment configuration: {cfg}")
    update_log_file(LOGGER, cfg.export_path / "experiment.log")
    export_config(cfg)
    main(cfg)
    end_time = datetime.now()
    LOGGER.info(f"Experiment completed. Results exported to: {cfg.export_path}")
    LOGGER.info(f"Total execution time: {end_time - start_time}")
