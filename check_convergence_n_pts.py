import tqdm
import itertools
import torch
# from finsler_embedding.experiment_run import *
from geodesic_toolbox.cometric import IdentityCoMetric, RandersMetrics,CoMetric

import pandas as pd
import seaborn as sns
from finsler_embedding.omega import VALID_OMEGA_TYPES


def sample_uniform2D(n_samples, bounds=[[-1, 1], [-1, 1]]):
    """
    Sample points uniformly in a 2D rectangle defined by bounds.
    """
    x_min, x_max = bounds[0]
    y_min, y_max = bounds[1]
    x_samples = torch.rand(n_samples) * (x_max - x_min) + x_min
    y_samples = torch.rand(n_samples) * (y_max - y_min) + y_min
    return torch.stack((x_samples, y_samples), dim=1)


def get_res_estimation(Lf_values, f_grad_values, v_true, c_km):
    """
    Compute the residual estimation for the operator approximation.
    """
    true_rhs = c_km * torch.einsum("n d, k n d -> k n", v_true, f_grad_values)  # (K, N)
    error_operator = (Lf_values - true_rhs).pow(2).mean(dim=0).sqrt()  # (N,)
    error_ratio = error_operator / (Lf_values.pow(2).mean(dim=0).sqrt() + 1e-8)  # (N,)
    return {
        "error_operator": error_operator.detach().numpy(),
        "error_ratio": error_ratio.detach().numpy(),
    }


def get_res_learning(Lf_values, f_grad_values, v_true, c_km, v_model):
    """
    Compute the residual learning for the vector field approximation.
    """
    true_rhs = c_km * torch.einsum("n d, k n d -> k n", v_true, f_grad_values)  # (K, N)
    model_rhs = c_km * torch.einsum("n d, k n d -> k n", v_model, f_grad_values)  # (K, N)
    error_learning = (Lf_values - model_rhs).pow(2).mean(dim=0).sqrt()  # (N,)
    error_wrt_v = (v_true - v_model).pow(2).mean(dim=1).sqrt()  # (N,)
    cosim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)(v_true, v_model)  # (N,)
    return {
        "error_learning": error_learning.detach().numpy(),
        "error_wrt_v": error_wrt_v.detach().numpy(),
        "cosim": cosim.detach().numpy(),
    }


def run_diagnostics(
    sample_function: callable,
    base_cometric: CoMetric,
    randers_metric: RandersMetrics,
    function_type: str,
    n_neighbors: int,
    m: int,
    K: int,
    N: int,
    tries_nb: int,
    theta=1.0,
):
    X = sample_function(N)

    edges = get_knn_graph(X, n_neighbors=n_neighbors, device="cpu")
    dst_edges = construct_distance_matrix(
        X, edges, randers_metric, use_approx=True, pbar=False
    )
    epsilon = compute_epsilon_empirical(dst_edges, scale=1)

    W, mu_0, mu_1 = gaussian_kernel(edges, dst_edges, epsilon, N, m)
    c_km = -mu_1 / mu_0 * (m + 1) / m

    Q_inv_theta = get_Q_inv_theta(W, theta=1)
    W_theta_s, W_theta_a = get_W_thetas(W, Q_inv_theta)
    P_theta_s, P_theta_a = construct_P_thetas(W_theta_s, W_theta_a)
    L_theta_s = 1 / epsilon**2 * P_theta_s
    L_theta_a = 1 / epsilon * P_theta_a

    f_values, Lf_values, f_grad_values = prepare_test_functions(K, function_type, X, L_theta_a)

    b_true = randers_metric.omega(X).detach() * randers_metric.beta
    v_true = b_to_v(b_true, base_cometric.cometric_tensor(X))

    v_model, omega_model, loss_list = train_vector_field_models(
        X, base_cometric, c_km, f_values, Lf_values, f_grad_values, pbar=False
    )

    res_estimation = get_res_estimation(Lf_values, f_grad_values, v_true, c_km)
    res_learning = get_res_learning(Lf_values, f_grad_values, v_true, c_km, v_model)

    new_df = {"N": N, "tries_nb": tries_nb} | res_estimation | res_learning
    return new_df


beta = 0.6
function_type = "coordinates"
n_neighbors = 10
m = 2
K = 500
N_values = [20, 50, 100, 200, 400, 800, 1600]
n_tries = 5

base_cometric = IdentityCoMetric()
omega_true = get_omega(omega_type="random", cometric=base_cometric)
randers_metric = RandersMetrics(omega=omega_true, base_cometric=base_cometric, beta=beta)

frames = []
pbar = tqdm.tqdm(
    total=len(N_values) * n_tries,
    desc="Sampling points and training models",
)
for N, tries_nb in pbar:
    new_df = run_diagnostics(
        sample_uniform2D,
        base_cometric,
        randers_metric,
        function_type,
        n_neighbors,
        m,
        K,
        N,
        tries_nb,
    )
    frames.append(pd.DataFrame(new_df))

res = pd.concat(frames)
