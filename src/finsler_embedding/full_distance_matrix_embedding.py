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

from finsler_embedding.experiment_run import *


def get_dense_graph_edges(X: torch.Tensor, device="cpu") -> torch.Tensor:
    """
    Constructs a dense graph by connecting every pair of points in X.

    Parameters:
    X : torch.Tensor (N, d)
        Input data points
    device : str
        Device to perform computations on

    Returns:
    edges : torch.Tensor (M, 2)
        Tensor containing pairs of indices representing edges in the dense graph
    """
    N = X.shape[0]
    # Create a grid of indices for all pairs (i, j)
    idx_i, idx_j = torch.meshgrid(torch.arange(N), torch.arange(N), indexing="ij")
    edges = torch.stack([idx_i.flatten(), idx_j.flatten()], dim=1).to(device)
    # edges = edges.to(torch.long)
    print(f"dtype of edges: {edges.dtype}, shape of edges: {edges.shape}")
    return edges


def instantiate_setup_dense(cfg: ExperimentConfig):
    assert cfg.omega_type == "constant", (
        "To be able to compute the full distance graph we use the"
        + "approximation which is exact for a conservative flow"
        + "we therefore require the omega_type to be 'constant'."
    )
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

    edges = get_dense_graph_edges(X, device=cfg.device)
    LOGGER.info(f"Found {edges.shape[0]} edges in the dense graph.")
    dst_edges = construct_distance_matrix(X, edges, randers_metric, use_approx=True, pbar=True)

    return X, bounds, base_cometric, randers_metric, edges, dst_edges


def main(cfg: ExperimentConfig):

    X, bounds, base_cometric, randers_metric, edges, dst_edges = instantiate_setup_dense(cfg)
    # X, bounds, base_cometric, randers_metric, edges, dst_edges = instantiate_setup(cfg)
    epsilon = compute_epsilon_nn(X)*5
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
        X, base_cometric, c_km, f_values, Lf_values, f_grad_values
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
