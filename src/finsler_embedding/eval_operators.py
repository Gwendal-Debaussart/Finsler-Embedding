import torch
from dataclasses import dataclass
import json
from pathlib import Path
import pandas as pd

from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import Isomap

from geodesic_toolbox import RandersMetrics, IdentityCoMetric, CentroidsCometric, SlopeMetrics

from finsler_embedding.graph import build_graph
from finsler_embedding.omega import *
from finsler_embedding.operators import randers_approximation, randers_constants


@dataclass
class EvalOperators:
    """Class to evaluate operators on a given model and dataset."""

    dataset: str  # Name of the dataset to use
    n: int  # Number of points in the dataset.
    m: int  # Dimension of the tangent space.
    beta: float  # Strength of the 1-form defining the Randers metric.
    graph_type: str  # Type of graph to build ('knn' or 'radius').
    embedding_type: str  # Type of embedding to use ('chart' or 'isomap').

    k: int  # Number of nearest neighbors for the graph.
    eps_scale: float  # Scale factor for the radius of the graph.

    eps0: float = 1.0  # Base radius for the graph (used if graph_type is 'radius').
    cut: float = 0.0  # Cutoff for the graph (used if graph_type is 'radius').
    radius: float = 1.0  # Radius for the graph (used if graph_type is 'radius').


@dataclass
class dataset_info:
    """Class to hold information about a dataset."""

    # High-dimensional data points (N, D)
    X_graph: torch.Tensor
    # Low-dimensional chart coordinates (N, m)
    X_chart: torch.Tensor
    # Tangent basis of the subspace where the Randers metric is restricted (N, D, m)
    P: torch.Tensor
    F: RandersMetrics
    # Centroid of the Randers metric restricted to the tangent spaces spanned by P, in tangent coordinates (N, m)
    c: torch.Tensor
    # Binet--Legendre cometric of the Randers metric restricted to the tangent spaces spanned by P, in tangent coordinates (N, m, m)
    g_bl_inv: torch.Tensor
    # Optional labels for the data points (N,)
    labels: torch.Tensor = None


def tangent_basis(X: torch.Tensor, m: int, k: int = 15) -> torch.Tensor:
    """
    Compute an orthonormal basis of the tangent spaces at each point in X using local PCA.

    Parameters:
    ----------
    X : torch.Tensor (N, D)
        Input data points.
    m : int
        Dimension of the tangent space.
    k : int, optional
        Number of nearest neighbors to use for local PCA (default is 15).

    Returns:
    -------
    V : torch.Tensor (N, D, m)
        Orthonormal basis of the tangent spaces at each point in X.
    """
    if X.shape[1] == m:
        return torch.eye(m).expand(X.shape[0], m, m).clone()
    idx = NearestNeighbors(n_neighbors=k).fit(X.numpy()).kneighbors(return_distance=False)
    nbrs = X[torch.from_numpy(idx)]  # (N, k, D)
    centered = nbrs - nbrs.mean(dim=1, keepdim=True)
    _, _, Vh = torch.linalg.svd(centered, full_matrices=False)
    return Vh[:, :m, :].transpose(1, 2)  # (N, D, m)


def diag_to_dense(A_inv: torch.Tensor) -> torch.Tensor:
    """Convert a diagonal cometric tensor to a dense cometric tensor."""
    if A_inv.ndim == 2:
        return torch.diag_embed(A_inv)
    elif A_inv.ndim == 3:
        return A_inv
    else:
        raise ValueError("A_inv must be either (N, D) or (N, D, D).")


def compute_unit_ball_info(
    A_inv: torch.Tensor, b: torch.Tensor, P: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the centroid c and the Binet--Legendre cometric g_bl_inv of the Randers metric
    restricted to the tangent spaces spanned by P, in tangent coordinates.

    Parameters:
    ----------
    A_inv : torch.Tensor (N, D) or (N, D, D)
        Ambient cometric tensor of the Randers metric.
    b : torch.Tensor (N, D)
        Ambient 1-form of the Randers metric.
    P : torch.Tensor (N, D, m)
        Tangent basis of the subspace where the Randers metric is restricted.

    Returns:
    -------
    c: torch.Tensor (N, m)
        Centroid of the Randers metric restricted to the tangent spaces spanned by P, in tangent coordinates.
    g_bl_inv: torch.Tensor (N, m, m)
        Binet--Legendre cometric of the Randers metric restricted to the tangent spaces sp
    """
    m = P.shape[2]

    A_inv = diag_to_dense(A_inv)
    A = torch.linalg.inv(A_inv)  # (N, D, D)
    A_T = P.transpose(1, 2) @ A @ P  # (N, m, m)
    A_T_inv = torch.linalg.inv(A_T)

    b_T = torch.einsum("ndm,nd->nm", P, b)  # (N, m)
    b_up = torch.einsum("nij,nj->ni", A_T_inv, b_T)  # A^{-1} b
    b_norm_sq = (b_T * b_up).sum(dim=1)  # ||b||^2_{A^{-1}}
    if (b_norm_sq >= 1).any():
        raise ValueError("The Randers metric is not well defined: ||b||_{A^-1} >= 1.")

    one_minus = (1 - b_norm_sq)[:, None]
    c = -b_up / one_minus

    C_up = A_T_inv + torch.einsum("ni,nj->nij", b_up, b_up) / one_minus[..., None]
    g_bl_inv = C_up / one_minus[..., None] + (m + 2) * torch.einsum("ni,nj->nij", c, c)

    return c, g_bl_inv


def empirical_unit_ball_info(
    F: RandersMetrics,
    X: torch.Tensor,
    n_theta: int = 720,
    batch: int = 200,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the empirical centroid and Binet--Legendre cometric of a Finsler metric F on a 2D chart.
    The computation is done by quadrature over the unit ball in polar coordinates. That is
    we compute the radius r(theta) = 1 / F(x, u_theta) for a set of angles theta, and then use the following formulas:
        |B| = 1/2 int r^2 dtheta
        c = 1/(3|B|) int r^3 u dtheta
        g_BL^{-1} = (m+2) S = 1/|B| int r^4 u u^T dtheta
    where u = (cos(theta), sin(theta)) is the unit vector in the direction of theta.

    Parameters:
    ----------
    F : RandersMetrics
        Finsler metric to evaluate.
    X : torch.Tensor (N, 2)
        Points at which to evaluate the Finsler metric.
    n_theta : int, optional
        Number of angles to use for the quadrature (default is 720).
    batch : int, optional
        Batch size for the computation (default is 200).

    Returns:
    -------
    c: torch.Tensor (N, 2)
        Empirical centroid of the Finsler metric at each point in X.
    g_bl_inv: torch.Tensor (N, 2, 2)
        Empirical Binet--Legendre cometric of the Finsler metric at each point
    """
    theta = torch.linspace(0, 2 * torch.pi, n_theta + 1)[:-1]
    u = torch.stack([theta.cos(), theta.sin()], dim=1)  # (T, 2)
    dth = 2 * torch.pi / n_theta
    cs, gs = [], []
    for start in range(0, X.shape[0], batch):
        x = X[start : start + batch]
        B = x.shape[0]
        xx = x[:, None, :].expand(B, n_theta, 2).reshape(-1, 2).clone().requires_grad_(True)
        uu = u[None].expand(B, n_theta, 2).reshape(-1, 2)
        r = (1 / F(xx, uu)).detach().reshape(B, n_theta)
        area = 0.5 * (r**2).sum(dim=1) * dth
        cs.append(torch.einsum("bt,ti->bi", r**3, u) * dth / (3 * area[:, None]))
        gs.append(torch.einsum("bt,ti,tj->bij", r**4, u, u) * dth / area[:, None, None])
    c = torch.cat(cs)
    g_bl_inv = torch.cat(gs)
    return c, g_bl_inv


def empirical_jacobian(
    X: torch.Tensor, Y: torch.Tensor, P: torch.Tensor, k: int = 15
) -> torch.Tensor:
    """
    Computes the Jacobian of an embedding Y with respect to the input X in the tangent coordinates defined by P
    using local least squares on the k-nearest neighbors.

    Parameters:
    ----------
    X : torch.Tensor (N, D)
        Input data points.
    Y : torch.Tensor (N, l)
        Embedded data points.
    P : torch.Tensor (N, D, m)
        Tangent basis of the subspace where the embedding is defined.
    k : int, optional
        Number of nearest neighbors to use for local least squares (default is 15).

    Returns:
    -------
    J : torch.Tensor (N, l, m)
        Jacobian of the embedding Y with respect to X in the tangent coordinates defined by P.
    """
    neighbors = NearestNeighbors(n_neighbors=k + 1).fit(X.numpy())
    idx = torch.from_numpy(
        neighbors.kneighbors(return_distance=False)[:, 1:]
    )  # Exclude self-neighbor
    dX = X[idx] - X[:, None, :]  # (N, k, D)
    dY = Y[idx] - Y[:, None, :]  # (N, k, l)
    dX_T = torch.einsum("nkd,ndm->nkm", dX, P)  # (N, k, m)
    sol = torch.linalg.lstsq(dX_T, dY).solution  # (N, m, l)
    return sol.transpose(1, 2)


class CachedCentroidsCometric(CentroidsCometric):
    """
    CentroidsCometric remembering its last cometric tensor evaluation.

    A Randers metric evaluates the cometric twice
    at the same points (for |v|_A and for the normalization of omega), and each evaluation sums
    over all the N centroids: the cache halves the cost of the Finsler distances.
    """

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        last = getattr(self, "_last", None)
        if last is not None and last[0] is z and last[1] == z._version:
            return last[2]
        out = super().forward(z)
        self._last = (z, z._version, out)
        return out


class Mountain(torch.nn.Module):
    """Height map of the Matsumoto experiment (matsumoto_dataset.ipynb), shape (N,)."""

    def forward(self, x):
        x1, x2 = x[:, 0], x[:, 1]
        return (
            0.4 * x1.sin() * x2.cos()
            + 0.15 * torch.exp(-((x1 + 1.0) ** 2 + (x2 - 0.5) ** 2))
            - 0.15 * torch.exp(-((x1 - 1.0) ** 2 + (x2 + 0.5) ** 2))
        )


def make_sphere(N: int, beta: float) -> dataset_info:
    """
    Make a dataset of points on the unit sphere in R^3, with a Randers metric defined by a circular 1-form.
    The tangent basis is computed analytically, and the ground truth distances are computed using the Randers metric.

    Parameters:
    ----------
    N : int
        Number of points to sample on the sphere.
    beta : float
        Strength of the 1-form defining the Randers metric.

    Returns:
    -------
    dataset_info
    """
    X = torch.nn.functional.normalize(torch.randn(N, 3), dim=1)
    cometric = IdentityCoMetric()
    omega = CircularSphereOmega(cometric, jitter=0.0)
    F = RandersMetrics(base_cometric=cometric, omega=omega, beta=beta)

    # Analytic tangent basis of the sphere at each point
    e = torch.tensor([0.0, 0.0, 1.0]).expand(N, 3)
    e_other = torch.tensor([0.0, 1.0, 0.0]).expand(N, 3)
    X_pole = (X[:, 2].abs() > 0.9)[:, None]
    e = torch.where(X_pole, e_other, e)
    t1 = torch.nn.functional.normalize(torch.linalg.cross(X, e), dim=1)  # (N, 3)
    t2 = torch.linalg.cross(X, t1)  # (N, 3)
    P = torch.stack([t1, t2], dim=2)  # (N, 3, 2)

    A_inv = cometric.cometric_tensor(X).detach()
    b = beta * omega(X).detach()
    c, g_bl_inv = compute_unit_ball_info(A_inv, b, P)

    return dataset_info(
        X_graph=X,
        X_chart=X,
        P=P,
        F=F,
        c=c,
        g_bl_inv=g_bl_inv,
        labels=None,
    )


def make_swiss_roll_data(N: int, beta: float, noise: float = 0.1) -> dataset_info:
    """
    Make a dataset of points on the Swiss roll in R^3, with a Randers metric defined by a 1-form.
    The tangent basis is computed using local PCA, and the ground truth distances are computed using the Randers metric.

    Parameters:
    ----------
    N : int
        Number of points to sample on the Swiss roll.
    beta : float
        Strength of the 1-form defining the Randers metric.
    noise : float
        Standard deviation of the Gaussian noise added to the Swiss roll.

    Returns:
    -------
    dataset_info
    """

    X, t = make_swiss_roll(n_samples=N, noise=noise)
    X = torch.from_numpy(X).float()
    cometric = CachedCentroidsCometric(
        centroids=X, cometric_centroids=IdentityCoMetric()(X), kappa=5
    )
    omega = OmegaSwissRoll(cometric, jitter=0.0)
    F = RandersMetrics(base_cometric=cometric, omega=omega, beta=beta)
    P = tangent_basis(X, m=2)

    A_inv = cometric.cometric_tensor(X).detach()
    b = beta * omega(X).detach()
    c, g_bl_inv = compute_unit_ball_info(A_inv, b, P)
    return dataset_info(
        X_graph=X,
        X_chart=X,
        P=P,
        F=F,
        c=c,
        g_bl_inv=g_bl_inv,
        labels=t,
    )


def make_matsumoto(N: int, beta: float = None) -> dataset_info:
    """
    Make a dataset of points on the Matsumoto mountain in R^3, with a Randers metric defined by a 1-form.
    The tangent basis is the standard basis of R^2, and the ground truth distances are computed using the Randers metric.

    Parameters:
    ----------
    N : int
        Number of points to sample on the Matsumoto mountain.
    beta : float
        Unused parameter, kept for compatibility with other dataset functions.

    Returns:
    -------
    dataset_info
    """
    x = torch.rand(N) * 6.4 - 3.2
    y = torch.rand(N) * 4.8 - 2.4
    X = torch.stack([x, y], dim=1)
    F = SlopeMetrics(Mountain())
    with torch.no_grad():
        X_high = torch.cat([X, Mountain()(X)[:, None]], dim=1)
    c, g_inv_bl = empirical_unit_ball_info(F, X)
    P = torch.eye(2).expand(N, 2, 2).clone()

    return dataset_info(
        X_graph=X_high,
        X_chart=X,
        P=P,
        F=F,
        c=c,
        g_inv_bl=g_inv_bl,
        labels=None,
    )


DATASETS = {
    "sphere": make_sphere,
    "swiss_roll": make_swiss_roll_data,
    "matsumoto": make_matsumoto,
}


def embed_data(data: dataset_info, config: EvalOperators):
    """
    Embed the data using either coordinates chart or isomap.
    Returns the low dimensional embedding of the data and the jacobian of the embedding.

    Parameters:
    ----------
    data : dataset_info
        The dataset information containing the high-dimensional points, tangent basis, and Randers metric.
    config : EvalOperators
        Configuration parameters for the evaluation, including graph type and parameters.

    Returns:
    -------
    Y : torch.Tensor (N, m)
        Low-dimensional embedding of the data.
    J : torch.Tensor (N, m, m)
        Jacobian of the embedding with respect to the input data in the tangent coordinates defined by P.
    """
    N, D, m = data.P.shape
    if config.embedding_type == "chart":
        Y = data.X_chart.detach().clone()
        # J = torch.eye(m).expand(N, m, m).clone()
        J = data.P  # ?
    elif config.embedding_type == "isomap":
        isomap = Isomap(n_neighbors=config.k, n_components=m)
        Y = torch.from_numpy(isomap.fit_transform(data.X_graph.numpy())).float()
        J = empirical_jacobian(data.X_graph, Y, data.P, k=config.k)
    else:
        raise ValueError(f"Unknown embedding type: {config.embedding_type}")

    return Y, J


def evaluate_embedding(
    res: dict,
    Y: torch.Tensor,
    J: torch.Tensor,
    data: dataset_info,
    m: int,
) -> dict:
    """Compare the estimated V, Gamma and ||c||_{g_BL} with the ground truth."""
    c_2, C, kappa = randers_constants(m)
    c, g_bl_inv = data.c, data.g_bl_inv
    g_bl = torch.linalg.inv(g_bl_inv)
    V_ref_all = C * torch.einsum("nlm,nm->nl", J, c)
    Gamma_ref = c_2 * J @ g_bl_inv @ J.transpose(1, 2)
    c_norm_ref = torch.einsum("ni,nij,nj->n", c, g_bl, c)

    # Keep only the admissible samples
    adm = res["admissible"]
    V, Gamma, c_norm = res["V"][adm], res["Gamma"][adm], res["c_norm_sq"][adm]
    V_ref, Gamma_ref, c_norm_ref = V_ref_all[adm], Gamma_ref[adm], c_norm_ref[adm]

    cos = torch.nn.functional.cosine_similarity(V, V_ref, dim=1)
    scale_V = (V * V_ref).sum(1) / (V_ref * V_ref).sum(1)
    relerr_V = (V - V_ref).norm(dim=1) / V_ref.norm(dim=1)
    trace_ratio_G = Gamma.diagonal(dim1=1, dim2=2).sum(1) / Gamma_ref.diagonal(
        dim1=1, dim2=2
    ).sum(1)
    relerr_G = (Gamma - Gamma_ref).flatten(1).norm(dim=1) / Gamma_ref.flatten(1).norm(dim=1)
    relerr_cnorm = (c_norm - c_norm_ref).abs() / c_norm_ref
    # Same error over all samples: restricting to the admissible ones selects the samples whose
    # (noisy) estimate is low, and biases the error when ||c||^2_{g_BL} is close to 1 / (m + 3)
    c_norm_all = torch.einsum("ni,nij,nj->n", c, g_bl, c)
    ratio_cnorm_all = res["c_norm_sq"] / c_norm_all
    relerr_c_norm_all = (res["c_norm_sq"] - c_norm_all).abs() / c_norm_all
    cos_all = torch.nn.functional.cosine_similarity(res["V"], V_ref_all, dim=1)

    def med(t: torch.Tensor) -> float:
        return t.median().item()

    return {
        "admissible": float(adm.float().mean()),
        "cos_V": med(cos),
        "scale_V": med(scale_V),
        "relerr_V": med(relerr_V),
        "trace_ratio_Gamma": med(trace_ratio_G),
        "relerr_Gamma": med(relerr_G),
        "relerr_c_norm": med(relerr_cnorm),
        "relerr_c_norm_all": med(relerr_c_norm_all),
        "ratio_c_norm_all": med(ratio_cnorm_all),
        "cos_V_all": med(cos_all),
    }


def export_dict(results_dict: dict, file_path: Path):
    """
    Export a dictionary to a JSON file.

    Parameters:
    ----------
    results_dict : dict
        The dictionary to export.
    file_path : Path
        The path to the JSON file where the dictionary will be saved.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    export_dict = results_dict.copy()
    for key, value in export_dict.items():
        if isinstance(value, torch.Tensor):
            export_dict[key] = value.detach().cpu().numpy().tolist()
        if isinstance(value, Path):
            export_dict[key] = str(value)
    with open(file_path, "w") as f:
        json.dump(export_dict, f, indent=4)


def main(config: EvalOperators):
    data: dataset_info = DATASETS[config.dataset](N=config.n, beta=config.beta)
    W, edges, dst, eps = build_graph(
        data.X_graph,
        data.F,
        graph_type=config.graph_type,
        k=config.k,
        eps_scale=config.eps_scale,
        eps0=config.eps0,
        m=config.m,
        cut=config.cut,
        radius=config.radius,
    )
    Y, J = embed_data(data, config)

    randers_approx = randers_approximation(
        W=W,
        epsilon=eps,
        X_low=Y,
        m=config.m,
        kernel_type="gaussian",
    )
    results_dict = evaluate_embedding(randers_approx, Y, J, data, config.m)
    df = pd.DataFrame([config.__dict__ | results_dict])
    return df


if __name__ == "__main__":
    config: EvalOperators = EvalOperators(
        dataset="swiss_roll",
        n=1000,
        m=2,
        beta=0.5,
        graph_type="radius",
        embedding_type="isomap",
        k=15,
        eps_scale=1.0,
        eps0=1.0,
        cut=0.0,
        radius=1.0,
    )

    df = main(config)
