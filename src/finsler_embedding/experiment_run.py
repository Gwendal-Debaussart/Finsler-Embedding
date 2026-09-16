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
from finsler_embedding.omega import get_omega, VALID_OMEGA_TYPES
from finsler_embedding.function_test import prepare_test_functions, VALID_TEST_FUNCTION_TYPES

LOGGER = setup_logger(name=None, log_file=None, level=logging.INFO)


def parse_args():
    parser = ArgumentParser(description="Run the experiment for estimating b from data.")
    parser.add_argument("--N", type=int, default=5000, help="Number of data points.")
    parser.add_argument("--m", type=int, default=2, help="Dimension of the manifold.")
    parser.add_argument(
        "--beta", type=float, default=0.5, help="Beta parameter for Randers metric."
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
    parser.add_argument(
        "--omega_type",
        type=str,
        default="round",
        choices=VALID_OMEGA_TYPES,
        help="Type of omega to use: 'round' or 'constant'.",
    )
    parser.add_argument(
        "--function_type",
        type=str,
        default="mexican",
        choices=VALID_TEST_FUNCTION_TYPES,
        help="Type of test functions to use: 'mexican', 'gaussian', 'mexican_and_gaussian', or 'coordinates'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the computations on (e.g., 'cpu' or 'cuda').",
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
        "--n_epochs",
        type=int,
        default=1000,
        help="Number of epochs for training the vector field model.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Epsilon parameter for the kernel. If None, it will be computed empirically.",
    )
    args = parser.parse_args()
    return args


@dataclass
class ExperimentConfig:
    N: int = 5000
    m: int = 2
    beta: float = 0.5
    K: int = 500
    n_neighbors: int = 10
    omega_type: str = "constant"
    function_type: str = "mexican_and_gaussian"
    device: str = "cpu"
    export_path: Path = Path("./results") / datetime.now().strftime("%Y%m%d_%H%M%S")
    no_plot: bool = False
    n_epochs: int = 1000
    epsilon: float = None


def sample_uniform2D(
    n_samples: int, bounds: list[list[float]] = [[-1, 1], [-1, 1]]
) -> torch.Tensor:
    """
    Sample points uniformly in a 2D rectangle defined by bounds.

    Parameters:
    -----------
    n_samples : int
        Number of samples to generate.
    bounds : list of list of float, optional
        Bounds for the sampling rectangle, default is [[-1, 1], [-1, 1]].

    Returns:
    --------
    X : torch.Tensor (n_samples, 2)
        Sampled points in 2D.
    """
    x_min, x_max, y_min, y_max = bounds
    x_samples = torch.rand(n_samples) * (x_max - x_min) + x_min
    y_samples = torch.rand(n_samples) * (y_max - y_min) + y_min
    return torch.stack((x_samples, y_samples), dim=1)


def sample_grid(
    n_samples: int, bounds: tuple = (-2, 2, -2, 2), device: str = "cpu"
) -> torch.Tensor:
    """
    Sample points on a 2D cartesian grid defined by bounds.
    """
    grid_size = int(np.sqrt(n_samples))
    x = np.linspace(bounds[0], bounds[1], grid_size)
    y = np.linspace(bounds[2], bounds[3], grid_size)
    xv, yv = np.meshgrid(x, y)
    samples = np.stack([xv.flatten(), yv.flatten()], axis=1)
    return torch.from_numpy(samples).float().to(device)


def export_config(cfg: ExperimentConfig):
    config_export = {}
    for field in cfg.__dataclass_fields__:
        value = getattr(cfg, field)
        if isinstance(value, Path):
            config_export[field] = str(value)
        else:
            config_export[field] = value

    with open(cfg.export_path / "experiment_config.json", "w") as f:
        json.dump(config_export, f, indent=4)

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
    X: torch.Tensor,
    base_cometric: CoMetric,
    c_km: torch.Tensor,
    f_values: torch.Tensor,
    Lf_values: torch.Tensor,
    f_grad_values: torch.Tensor,
    device: str = "cpu",
    m: int = 2,
    n_epochs: int = 1000,
    pbar: bool = True,
) -> tuple[torch.nn.Module, torch.nn.Module, list[float]]:
    v_model = V_estimator(hidden_dims=[64, 64], dim=m).to(device)
    omega_model = OmegaFromV(v_estimator=v_model, cometric=base_cometric).to(device)
    loss_list = learn_v(
        X,
        f_values,
        Lf_values,
        f_grad_values,
        c_km,
        v_model,
        device=device,
        n_epochs=n_epochs,
        pbar=pbar,
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


def instantiate_setup(cfg: ExperimentConfig):
    X = sample_uniform2D(cfg.N, bounds=(-2, 2, -2, 2))
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
    dst_edges = construct_distance_matrix(
        X, edges, randers_metric, use_approx=True, pbar=False
    )

    return X, bounds, base_cometric, randers_metric, edges, dst_edges


def prepare_operators(
    cfg: ExperimentConfig,
    edges: torch.Tensor,
    dst_edges: torch.Tensor,
    epsilon: float,
    operator_type: str = "gaussian",
    theta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Prepare the operators L_theta_s and L_theta_a based on the specified operator type.

    Parameters:
    -----------
    cfg : ExperimentConfig
        The configuration of the experiment.
    edges : torch.Tensor
        The edges of the KNN graph.
    dst_edges : torch.Tensor
        The distances corresponding to the edges of the KNN graph.
    epsilon : float
        The bandwidth parameter for the kernel.
    operator_type : str, optional
        The type of operator to use: 'gaussian' or 'laplacian'. Default is 'gaussian'.
    theta : float, optional
        The parameter for the theta operator. Default is 1.0.

    Returns:
    --------
    c_km : torch.Tensor
        The constant c_km used in the operators.
    L_theta_s : torch.Tensor
        The symmetric operator L_theta_s.
    L_theta_a : torch.Tensor
        The antisymmetric operator L_theta_a.
    W_theta_s : torch.Tensor
        The symmetric weight matrix W_theta_s.
    """
    LOGGER.info("Computing W, mu_0, mu_1, and operators L_theta_s and L_theta_a...")
    if operator_type == "gaussian":
        W, mu_0, mu_1 = gaussian_kernel(edges, dst_edges, epsilon, cfg.N, cfg.m)
    elif operator_type == "laplacian":
        W, mu_0, mu_1 = laplacian_kernel(edges, dst_edges, epsilon, cfg.N, cfg.m)
    else:
        raise ValueError(
            f"Unknown operator_type: {operator_type}. Must be 'gaussian' or 'laplacian'."
        )
    c_km = -mu_1 / mu_0 * (cfg.m + 1) / cfg.m

    Q_inv_theta = get_Q_inv_theta(W, theta=theta)
    W_theta_s, W_theta_a = get_W_thetas(W, Q_inv_theta)
    P_theta_s, P_theta_a = construct_P_thetas(W_theta_s, W_theta_a)
    L_theta_s = 1 / epsilon**2 * P_theta_s
    L_theta_a = 1 / epsilon * P_theta_a

    assert L_theta_a.isfinite().all(), "L_theta_a contains NaN or Inf values."
    assert L_theta_s.isfinite().all(), "L_theta_s contains NaN or Inf values."
    return c_km, L_theta_s, L_theta_a, W_theta_s


def check_constant_function(cfg: ExperimentConfig, X: torch.Tensor, L_theta_a: torch.Tensor):
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
    return res_LF_value_constant, Lf_values_constant


def check_true_value_delta(true_rhs: torch.Tensor, Lf_values: torch.Tensor):
    delta_rhs = true_rhs - Lf_values  # (K, N)
    delta_rhs_norm = torch.norm(delta_rhs, dim=0)  # (N,)
    res_delta_rhs_norm = qqt_summary_statistics(delta_rhs_norm)
    LOGGER.info(
        r"Delta_rhs_norm = $\| Lf_k(x_i) - c_km <v(x_i), nabla f_k(x_i)>\|$ (averaged over K)"
    )
    LOGGER.info(
        f"Delta_rhs_norm: mean = {res_delta_rhs_norm['mean']:.2e}, std = {res_delta_rhs_norm['std']:.2e}, min = {res_delta_rhs_norm['min']:.2e}, max = {res_delta_rhs_norm['max']:.2e}"
    )
    return res_delta_rhs_norm, delta_rhs


def check_true_value_ratio(true_rhs: torch.Tensor, Lf_values: torch.Tensor):
    ratio_rhs = true_rhs / (Lf_values + 1e-8)  # Avoid division by zero
    ratio_rhs_mean = ratio_rhs.mean(dim=1)
    res_ratio_rhs_mean = qqt_summary_statistics(ratio_rhs_mean)
    LOGGER.info(f"Ratio_rhs_mean should be 1")
    LOGGER.info(
        f"Ratio_rhs_mean: mean = {res_ratio_rhs_mean['mean']:.2e}, std = {res_ratio_rhs_mean['std']:.2e}, min = {res_ratio_rhs_mean['min']:.2e}, max = {res_ratio_rhs_mean['max']:.2e}"
    )
    return res_ratio_rhs_mean, ratio_rhs_mean


def check_cosim_estimate(b_hat_deep: torch.Tensor, b_true: torch.Tensor):
    cosim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    cosine_similarity_deep = cosim(b_hat_deep, b_true)
    res_cosine_similarity_deep = qqt_summary_statistics(cosine_similarity_deep)
    LOGGER.info(f"Cosine similarity between true b and estimated b (Deep Model) should be 1")
    LOGGER.info(
        f"Cosine similarity: mean = {res_cosine_similarity_deep['mean']:.2e}, std = {res_cosine_similarity_deep['std']:.2e}, min = {res_cosine_similarity_deep['min']:.2e}, max = {res_cosine_similarity_deep['max']:.2e}"
    )
    return res_cosine_similarity_deep, cosine_similarity_deep


def check_alpha_ratio(v_hat_deep: torch.Tensor, v_true: torch.Tensor):
    # We regress cst such that v_hat_deep = cst * v_true. We should have cst = 1 if the estimation is perfect.
    v_true_norm = v_true.norm(dim=1)
    alpha_ratio = torch.einsum("n d, n d -> n", v_hat_deep, v_true) / (
        v_true_norm**2 + 1e-8
    )  # Avoid division by zero
    res_alpha_ratio = qqt_summary_statistics(alpha_ratio)
    LOGGER.info(f"Alpha ratio should be 1")
    LOGGER.info(
        f"Alpha ratio: mean = {res_alpha_ratio['mean']:.2e}, std = {res_alpha_ratio['std']:.2e}, min = {res_alpha_ratio['min']:.2e}, max = {res_alpha_ratio['max']:.2e}"
    )
    return res_alpha_ratio, alpha_ratio


def check_v_est_vs_v_true(v_hat_deep: torch.Tensor, v_true: torch.Tensor):
    """
    Check the estimation of the vector field v_hat_deep against the true vector field v_true.
    """
    error_wrt_v = (v_true - v_hat_deep).pow(2).mean(dim=1).sqrt()  # (N,)
    res_error_wrt_v = qqt_summary_statistics(error_wrt_v)
    LOGGER.info(f"Error between true v and estimated v (Deep Model)")
    LOGGER.info(
        f"Error: mean = {res_error_wrt_v['mean']:.2e}, std = {res_error_wrt_v['std']:.2e}, min = {res_error_wrt_v['min']:.2e}, max = {res_error_wrt_v['max']:.2e}"
    )
    return res_error_wrt_v, error_wrt_v


def plot_all_results(
    cfg,
    base_cometric,
    X,
    randers_metric,
    bounds,
    loss_list,
    b_true,
    b_hat_deep,
    Lf_values_constant,
    delta_rhs,
    ratio_rhs_mean,
    cosine_similarity_deep,
    alpha_ratio,
    error_wrt_v,
):
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
        Lf_values_constant.flatten().cpu().detach().numpy(), bins=20, color="blue", alpha=0.7
    )
    ax.set_title("Distribution of $L[1](x_i)$ ")
    ax.vlines(
        x=0,
        ymin=0,
        ymax=ax.get_ylim()[1],
        colors="red",
        linestyles="dashed",
        label="y=0",
    )
    fig = ax.get_figure()
    fig.savefig(cfg.export_path / "Lf_values_constant_distribution.png", dpi=300)
    plt.close()

    LOGGER.info("Plotting the distribution of delta_rhs...")
    sns.histplot(delta_rhs.flatten().cpu().detach().numpy(), bins=20, color="blue", alpha=0.7)
    plt.gca().set_yscale("log")
    plt.title(r"Distribution of $ Lf_k(x_i) - c_km <v(x_i), \nabla f_k(x_i)>$")
    plt.savefig(cfg.export_path / "delta_rhs_distribution.png", dpi=300)
    plt.close()

    LOGGER.info("Plotting the distribution of ratio_rhs_mean...")
    sns.histplot(
        ratio_rhs_mean.flatten().cpu().detach().numpy(), bins=20, color="blue", alpha=0.7
    )
    plt.gca().set_yscale("log")
    plt.title(r"Distribution of $\frac{Lf_k(x_i)}{c_km <v(x_i), \nabla f_k(x_i)>}$")
    plt.savefig(cfg.export_path / "ratio_rhs_distribution.png", dpi=300)
    plt.close()

    LOGGER.info(
        "Plotting the distribution of cosine similarity between true b and estimated b (Deep Model)..."
    )
    plt.hist(cosine_similarity_deep.cpu().detach().numpy(), bins=50, color="blue", alpha=0.7)
    plt.title("Cosine Similarity between True and Estimated b (Deep Model)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.savefig(cfg.export_path / "cosine_similarity_deep.png", dpi=300)
    plt.close()

    LOGGER.info("Plotting the distribution of alpha ratio...")
    plt.hist(alpha_ratio.cpu().detach().numpy(), bins=50, color="blue", alpha=0.7)
    plt.xlabel("Alpha Ratio")
    plt.ylabel("Frequency")
    plt.title(
        r"Distribution of $\alpha = \frac{<v_{hat}(x_i), v_{true}(x_i)>}{\|v_{true}(x_i)\|^2}$"
    )
    plt.savefig(cfg.export_path / "alpha_ratio_distribution.png", dpi=300)
    plt.close()

    LOGGER.info(
        "Plotting the distribution of error between true v and estimated v (Deep Model)..."
    )
    plt.hist(error_wrt_v.cpu().detach().numpy(), bins=50, color="blue", alpha=0.7)
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.title(r"Distribution of $\|v_{hat}(x_i) - v_{true}(x_i)\|_2$")
    plt.savefig(cfg.export_path / "error_wrt_v_distribution.png", dpi=300)
    plt.close()


def main(cfg: ExperimentConfig):

    X, bounds, base_cometric, randers_metric, edges, dst_edges = instantiate_setup(cfg)
    if cfg.epsilon is not None:
        epsilon = cfg.epsilon
    else:
        epsilon = compute_epsilon_empirical(dst_edges, scale=2.0)
    c_km, L_theta_s, L_theta_a, W_theta_s = prepare_operators(cfg, edges, dst_edges, epsilon)
    f_values, Lf_values, f_grad_values = prepare_test_functions(
        cfg.K, cfg.function_type, X, L_theta_a
    )

    # Compute the true vector field v_true and the corresponding b_true
    b_true = randers_metric.omega(X) * randers_metric.beta
    v_true = b_to_v(b_true, base_cometric.cometric_tensor(X)).detach()
    true_rhs = c_km * torch.einsum("n d, k n d -> k n", v_true, f_grad_values)

    LOGGER.info("Training vector field models...")
    v_model, omega_model, loss_list = train_vector_field_models(
        X,
        base_cometric,
        c_km,
        f_values,
        Lf_values,
        f_grad_values,
        n_epochs=cfg.n_epochs,
    )
    v_hat_deep = v_model(X).detach()
    b_hat_deep = omega_model(X).detach()

    # Analyze results
    res_LF_value_constant, Lf_values_constant = check_constant_function(cfg, X, L_theta_a)
    res_delta_rhs_norm, delta_rhs = check_true_value_delta(true_rhs, Lf_values)
    res_ratio_rhs_mean, ratio_rhs_mean = check_true_value_ratio(true_rhs, Lf_values)
    res_cosine_similarity_deep, cosine_similarity_deep = check_cosim_estimate(
        b_hat_deep, b_true
    )
    res_alpha_ratio, alpha_ratio = check_alpha_ratio(v_hat_deep, v_true)
    res_error_wrt_v, error_wrt_v = check_v_est_vs_v_true(v_hat_deep, v_true)

    all_results = {
        "res_LF_value_constant": res_LF_value_constant,
        "res_delta_rhs_norm": res_delta_rhs_norm,
        "res_ratio_rhs_mean": res_ratio_rhs_mean,
        "res_cosine_similarity_deep": res_cosine_similarity_deep,
        "res_alpha_ratio": res_alpha_ratio,
        "res_error_wrt_v": res_error_wrt_v,
    }
    export_results(cfg, all_results)

    ######################################
    # Plot the training loss and other distributions
    ######################################
    if cfg.no_plot:
        LOGGER.info("Plotting is disabled. Skipping plot generation.")
    else:
        plot_all_results(
            cfg,
            base_cometric,
            X,
            randers_metric,
            bounds,
            loss_list,
            b_true,
            b_hat_deep,
            Lf_values_constant,
            delta_rhs,
            ratio_rhs_mean,
            cosine_similarity_deep,
            alpha_ratio,
            error_wrt_v,
        )


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
        function_type=args.function_type,
        n_epochs=args.n_epochs,
        epsilon=args.epsilon,
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
