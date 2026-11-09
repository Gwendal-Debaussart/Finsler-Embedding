"""
Quantitative evaluation of the moment-Randers embedding (Algorithm 1 of the paper).

For each dataset, the graph operators L^s and L^a are built from sampled points and compared
with the ground truth given by the continuous theory:

    V_i     = (L^a Psi)_i            ~  C   J c(X_i)                 (Thm. sym_limit)
    Gamma_i = carre du champ of L^s  ~  c_2 J g_BL^{-1}(X_i) J^T     (Prop. carre_du_champ)
    kappa V_i^T Gamma_i^+ V_i        ~  ||c(X_i)||^2_{g_BL}          (embedding independent)

where c is the centroid of the unit ball, g_BL the Binet--Legendre metric and J the Jacobian of
the embedding Psi. The ground truth (c, g_BL) is given in closed form for Randers metrics
(Prop. centroid_randers_metric) and computed by quadrature over the unit ball otherwise.

Datasets
--------
sphere      Randers metric on the unit sphere S^2 (no boundary), rotational drift.
swiss_roll  Randers metric on the Swiss roll, drift tangent to the roll.
matsumoto   Slope (Matsumoto) metric of a height map, a non-Randers metric.
disbm       Directed stochastic block model: direction of the drift vs the flow between communities.

Usage
-----
    python -m finsler_embedding.evaluate sphere --seeds 5
    python -m finsler_embedding.evaluate sphere --n-list 250 500 1000 2000 4000 --graph radius
    python -m finsler_embedding.evaluate swiss_roll --embedding chart isomap
    python -m finsler_embedding.evaluate matsumoto
    python -m finsler_embedding.evaluate disbm --p-list 0.25 0.35 0.4 0.45 0.5 0.55 0.65

Results are written to <out>/<dataset>_<graph parameters>.csv (one row per run), and a summary
(median ± interquartile range over seeds) is printed.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.datasets import make_swiss_roll
from sklearn.neighbors import NearestNeighbors

from geodesic_toolbox.cometric import (
    IdentityCoMetric,
    RandersMetrics,
    SlopeMetrics,
)
from finsler_embedding.experiments.disbm.disbm import (
    DiSBMConfig,
)
from finsler_embedding.experiments.disbm.embedding import EmbeddingConfig, run_direction
from finsler_embedding.eval_operators import (
    CachedCentroidsCometric,
    Mountain,
    compute_unit_ball_info,
    empirical_unit_ball_info,
    tangent_basis,
)
from finsler_embedding.omega import CircularSphereOmega, OmegaSwissRoll
from finsler_embedding.experiments.graph import build_graph as _build_graph
from finsler_embedding.core.embeddings import isomap_embedding
from finsler_embedding.core.kernels import randers_constants
from finsler_embedding.core.operators import randers_approximation

logging.getLogger().setLevel(logging.WARNING)


##############################
# Ground truth
##############################


def randers_truth(A_inv: torch.Tensor, b: torch.Tensor, P: torch.Tensor) -> dict:
    """Return Randers ground truth using the shared unit-ball implementation."""
    c, g_bl_inv = compute_unit_ball_info(A_inv, b, P)
    return {"c": c, "g_bl_inv": g_bl_inv}


def quadrature_truth(F, X: torch.Tensor, n_theta: int = 720, batch: int = 200) -> dict:
    """Return numerical ground truth using the shared unit-ball implementation."""
    c, g_bl_inv = empirical_unit_ball_info(F, X, n_theta=n_theta, batch=batch)
    return {"c": c, "g_bl_inv": g_bl_inv}


def local_jacobian(X: torch.Tensor, Y: torch.Tensor, P: torch.Tensor, k: int = 15) -> torch.Tensor:
    """
    Jacobian (N, l, m) of an embedding Y = Psi(X) in the tangent coordinates given by P,
    by least squares on the neighbors: Y_j - Y_i ~ J_i P_i^T (X_j - X_i).
    """
    idx = torch.from_numpy(
        NearestNeighbors(n_neighbors=k + 1).fit(X.numpy()).kneighbors(return_distance=False)[:, 1:]
    )
    dX = X[idx] - X[:, None, :]  # (N, k, D)
    dY = Y[idx] - Y[:, None, :]  # (N, k, l)
    dX_T = torch.einsum("nkd,ndm->nkm", dX, P)  # (N, k, m)
    sol = torch.linalg.lstsq(dX_T, dY).solution  # (N, m, l)
    return sol.transpose(1, 2)


##############################
# Estimation and metrics
##############################


def build_graph(X_graph, F, graph: str, k: int, eps_scale: float, eps0: float, m: int, cut: float = 3.0,
                radius: float = None, device: str = "cpu"):
    """Compatibility wrapper around the shared experiment graph builder."""
    if device != "cpu":
        raise ValueError("the shared graph builder supports CPU distances only")
    return _build_graph(
        X_graph,
        F,
        graph_type=graph,
        k=k,
        eps_scale=eps_scale,
        eps0=eps0,
        m=m,
        cut=cut,
        radius=radius,
    )


def estimated_F(res: dict, Y: torch.Tensor, i: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Embedded moment-Randers metric hat F_R(Y_i, v) (Def. embedded_moment_randers)."""
    H, c, lam = res.cometric[i], res.centroid[i], res.lambda_[i]
    v_H = torch.einsum("bij,bi,bj->b", H, v, v)
    c_H = torch.einsum("bij,bi,bj->b", H, c, v)
    return ((lam * v_H + c_H**2).clamp_min(0).sqrt() - c_H) / lam


def edge_asymmetry_spearman(res, Y, edges, dst, N) -> float:
    """Spearman correlation between true and estimated asymmetry ratios of the edges."""
    key = edges[:, 0] * N + edges[:, 1]
    lookup = dict(zip(key.tolist(), dst.tolist()))
    fwd = edges[edges[:, 0] < edges[:, 1]]
    back_key = (fwd[:, 1] * N + fwd[:, 0]).tolist()
    has_back = torch.tensor([kk in lookup for kk in back_key])
    fwd = fwd[has_back]
    adm = res.admissible
    fwd = fwd[adm[fwd[:, 0]] & adm[fwd[:, 1]]]
    if fwd.shape[0] < 10:
        return float("nan")
    i, j = fwd[:, 0], fwd[:, 1]
    d_ij = torch.tensor([lookup[a] for a in (i * N + j).tolist()])
    d_ji = torch.tensor([lookup[a] for a in (j * N + i).tolist()])
    F_ij = estimated_F(res, Y, i, Y[j] - Y[i])
    F_ji = estimated_F(res, Y, j, Y[i] - Y[j])
    ok = (F_ij > 0) & (F_ji > 0)
    return float(spearmanr((d_ij / d_ji)[ok].log(), (F_ij / F_ji)[ok].log()).statistic)


def evaluate_embedding(res, Y, J, truth, m, extra) -> dict:
    """Compare the estimated V, Gamma and ||c||_{g_BL} with the ground truth."""
    c_2, C, kappa = randers_constants(m)
    c, g_bl_inv = truth["c"], truth["g_bl_inv"]
    V_ref = C * torch.einsum("nlm,nm->nl", J, c)
    G_ref = c_2 * J @ g_bl_inv @ J.transpose(1, 2)
    c_norm_true = torch.einsum("ni,nij,nj->n", c, torch.linalg.inv(g_bl_inv), c)

    adm = res.admissible
    V, G = res.drift[adm], res.carré_du_champ[adm]
    V_ref, G_ref, c_norm_true = V_ref[adm], G_ref[adm], c_norm_true[adm]

    cos = torch.nn.functional.cosine_similarity(V, V_ref, dim=1)
    scale_V = (V * V_ref).sum(1) / (V_ref * V_ref).sum(1)
    relerr_V = (V - V_ref).norm(dim=1) / V_ref.norm(dim=1)
    trace_ratio_G = G.diagonal(dim1=1, dim2=2).sum(1) / G_ref.diagonal(dim1=1, dim2=2).sum(1)
    relerr_G = (G - G_ref).flatten(1).norm(dim=1) / G_ref.flatten(1).norm(dim=1)
    relerr_cnorm = (res.drift_norm_squared[adm] - c_norm_true).abs() / c_norm_true
    # Same error over all samples: restricting to the admissible ones selects the samples whose
    # (noisy) estimate is low, and biases the error when ||c||^2_{g_BL} is close to 1 / (m + 3)
    c_norm_all = torch.einsum("ni,nij,nj->n", c, torch.linalg.inv(g_bl_inv), c)
    ratio_cnorm_all = res.drift_norm_squared / c_norm_all
    cos_all = torch.nn.functional.cosine_similarity(res.drift, C * torch.einsum("nlm,nm->nl", J, c), dim=1)

    def med(t):
        return float(t.median())

    return extra | {
        "admissible": float(adm.float().mean()),
        "cos_V": med(cos),
        "cos_V_q10": float(cos.quantile(0.1)),
        "scale_V": med(scale_V),
        "relerr_V": med(relerr_V),
        "trace_ratio_Gamma": med(trace_ratio_G),
        "relerr_Gamma": med(relerr_G),
        "relerr_c_norm": med(relerr_cnorm),
        "relerr_c_norm_all": med((ratio_cnorm_all - 1).abs()),
        "ratio_c_norm_all": med(ratio_cnorm_all),
        "cos_V_all": med(cos_all),
    }


##############################
# Datasets
##############################


def make_sphere(N: int, beta: float):
    X = torch.nn.functional.normalize(torch.randn(N, 3), dim=1)
    cometric = IdentityCoMetric()
    omega = CircularSphereOmega(cometric, jitter=0.0)
    F = RandersMetrics(base_cometric=cometric, omega=omega, beta=beta)
    # Analytic tangent basis of the sphere
    e = torch.tensor([0.0, 0.0, 1.0]).expand(N, 3)
    e = torch.where((X[:, 2].abs() > 0.9)[:, None], torch.tensor([1.0, 0.0, 0.0]).expand(N, 3), e)
    t1 = torch.nn.functional.normalize(torch.linalg.cross(X, e), dim=1)
    t2 = torch.linalg.cross(X, t1)
    P = torch.stack([t1, t2], dim=2)
    truth = randers_truth(cometric.cometric_tensor(X), beta * omega(X), P)
    return {"X_graph": X, "X_chart": X, "P": P, "F": F, "truth": truth, "labels": None}


def make_swiss_roll_data(N: int, beta: float, noise: float = 0.1):
    X, t = make_swiss_roll(n_samples=N, noise=noise)
    X = torch.from_numpy(X).float()
    # Same cometric as the paper figure (kappa = 5)
    cometric = CachedCentroidsCometric(centroids=X, cometric_centroids=IdentityCoMetric()(X), kappa=5)
    omega = OmegaSwissRoll(cometric, jitter=0.0)
    F = RandersMetrics(base_cometric=cometric, omega=omega, beta=beta)
    P = tangent_basis(X, m=2)
    truth = randers_truth(cometric.cometric_tensor(X).detach(), beta * omega(X).detach(), P)
    return {"X_graph": X, "X_chart": X, "P": P, "F": F, "truth": truth, "labels": t}


def make_matsumoto(N: int, beta: float = None):
    X = torch.stack([torch.rand(N) * 6.4 - 3.2, torch.rand(N) * 4.8 - 2.4], dim=1)
    F = SlopeMetrics(Mountain())
    with torch.no_grad():
        X_high = torch.cat([X, Mountain()(X)[:, None]], dim=1)
    truth = quadrature_truth(F, X)
    P = torch.eye(2).expand(N, 2, 2).clone()
    return {"X_graph": X_high, "X_chart": X, "P": P, "F": F, "truth": truth, "labels": None}


DATASETS = {"sphere": make_sphere, "swiss_roll": make_swiss_roll_data, "matsumoto": make_matsumoto}


def run_manifold(args) -> pd.DataFrame:
    rows = []
    m = 2
    n_list = args.n_list or [args.n]
    for N in n_list:
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            data = DATASETS[args.dataset](N, args.beta)
            W, edges, dst, eps = build_graph(
                data["X_graph"], data["F"], args.graph, args.k, args.eps_scale, args.eps0, m, args.cut,
                args.radius, args.device,
            )
            for emb in args.embedding:
                if emb == "chart":
                    # Psi = ambient (or chart) coordinates: J is the inclusion of the tangent space
                    Y = data["X_chart"].detach()
                    J = data["P"]
                elif emb == "isomap":
                    Y = isomap_embedding(data["X_graph"].detach(), n_components=m, n_neighbors=10)
                    J = local_jacobian(data["X_chart"].detach(), Y, data["P"])
                else:
                    raise ValueError(f"unknown embedding {emb!r}")
                res = randers_approximation(W, eps, X_low=Y, m=m)
                row = evaluate_embedding(
                    res, Y, J, data["truth"], m,
                    {"dataset": args.dataset, "embedding": emb, "N": N, "seed": seed, "beta": args.beta,
                     "graph": args.graph, "k": args.k, "eps_scale": args.eps_scale, "eps0": args.eps0, "cut": args.cut, "radius": args.radius,
                     "eps": eps, "n_edges": edges.shape[0]},
                )
                row["spearman_edge_ratio"] = edge_asymmetry_spearman(res, Y, edges, dst, N)
                rows.append(row)
                print(
                    f"{args.dataset} {emb:6s} N={N:5d} seed={seed}: cos_V={row['cos_V']:.3f} "
                    f"scale_V={row['scale_V']:.2f} relerr_Gamma={row['relerr_Gamma']:.2f} "
                    f"relerr_c_norm={row['relerr_c_norm']:.2f} spearman={row['spearman_edge_ratio']:.2f} "
                    f"admissible={row['admissible']:.2f}",
                    flush=True,
                )
    return pd.DataFrame(rows)


##############################
# DiSBM
##############################


def disbm_direction(p: float, r: float, K: int, N: int, dim: int = 2, eps: float = 1.0) -> dict:
    """
    Direction of the recovered drift on one DiSBM graph. The ground truth is the flow between
    communities: forward (k -> k+1) if p > q, backward if q > p, none if p = q. With mu_k the center
    of community k in the embedding, the forward direction at community k is the tangent to the cycle
    of communities, mu_{k+1} - mu_{k-1} (the chord mu_{k+1} - mu_k is off by half the angle between
    two consecutive communities).

    Returns the signed cosines with the forward direction (> 0 forward, < 0 backward, ~ 0 without
    flow) of the community-mean drifts (K,) and of the node drifts (N,), and the node values of
    ||hat c||_{hat g_BL}, the estimated strength of the asymmetry.
    """
    result = run_direction(
        p,
        DiSBMConfig(n=N, communities=K, r=r),
        EmbeddingConfig(embedding_dim=dim, bandwidth=eps),
    )
    return {
        "q": result.q,
        "cos_community": result.cos_community,
        "cos_node": result.cos_node,
        "c_norm": result.c_norm,
        "admissible": result.admissible,
    }


def run_disbm(args) -> pd.DataFrame:
    """
    Direction metrics on a DiSBM (see `disbm_direction`), for each p in args.p_list:
      cos_forward_*   signed cosine with the forward direction (community means / nodes, median)
      flow_cos_*      cosine with the direction of the dominant flow (undefined when p = q)
      flow_accuracy_* fraction of communities / nodes whose drift points along the dominant flow
      c_norm          median of ||hat c||_{hat g_BL}
    """
    rows = []
    K, N, r = args.communities, args.n_disbm, args.r
    for p in args.p_list:
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)
            out = disbm_direction(p, r, K, N, args.dim)
            q, cos_comm, cos_node = out["q"], out["cos_community"], out["cos_node"]
            row = {"dataset": "disbm", "N": N, "K": K, "p": p, "q": q, "r": r, "p_minus_q": p - q,
                   "seed": seed, "dim": args.dim,
                   "admissible": float(out["admissible"].float().mean()),
                   "cos_forward_community": float(cos_comm.mean()),
                   "cos_forward_node": float(cos_node.median()),
                   "c_norm": float(out["c_norm"].median())}
            if abs(p - q) > 1e-9:
                sign = 1.0 if p > q else -1.0
                row |= {"flow_cos_community": float((sign * cos_comm).mean()),
                        "flow_cos_community_min": float((sign * cos_comm).min()),
                        "flow_accuracy_community": float((sign * cos_comm > 0).float().mean()),
                        "flow_cos_node": float((sign * cos_node).median()),
                        "flow_accuracy_node": float((sign * cos_node > 0).float().mean())}
            rows.append(row)
            print(
                f"disbm p={p:.2f} q={q:.2f} seed={seed}: cos_forward community={row['cos_forward_community']:+.3f} "
                f"node={row['cos_forward_node']:+.3f}, "
                f"node accuracy={row.get('flow_accuracy_node', float('nan')):.3f}, "
                f"||c||={row['c_norm']:.3f}",
                flush=True,
            )
    return pd.DataFrame(rows)


##############################
# Main
##############################


def summarize(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    params = ["eps", "beta", "eps_scale", "eps0", "cut", "p", "q", "r", "p_minus_q"]
    metrics = [c for c in df.columns if df[c].dtype.kind == "f" and c not in by + params]
    g = df.groupby(by)[metrics]
    return g.median().round(3).astype(str) + " ± " + (g.quantile(0.75) - g.quantile(0.25)).round(3).astype(str)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=list(DATASETS) + ["disbm"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("results/evaluation"))
    # Manifold datasets
    parser.add_argument("--n", type=int, default=2000, help="Number of samples.")
    parser.add_argument("--n-list", type=int, nargs="+", help="Sweep over N (convergence study).")
    parser.add_argument("--beta", type=float, default=0.5, help="||b||_{A^-1} of the Randers drift.")
    parser.add_argument("--graph", choices=["knn", "radius"], default="knn")
    parser.add_argument("--k", type=int, default=10, help="Neighbors of the kNN graph.")
    parser.add_argument("--eps-scale", type=float, default=1.0, help="kNN graph: eps = scale * std(d).")
    parser.add_argument("--eps0", type=float, default=0.5, help="Radius graph: eps = eps0 (log N / N)^(1/(m+4)).")
    parser.add_argument("--cut", type=float, default=3.0,
                        help="Radius graph: Euclidean cutoff in units of eps (>= 3 / (1 - ||b||) for a Randers metric).")
    parser.add_argument("--radius", type=float, default=None,
                        help="Radius graph with this fixed Euclidean radius, eps chosen so that the kernel is not truncated.")
    parser.add_argument("--device", default="cpu", help="Device for the Finsler distances (cpu, mps, cuda).")
    parser.add_argument("--embedding", nargs="+", default=["chart", "isomap"], choices=["chart", "isomap"])
    # DiSBM
    parser.add_argument("--n-disbm", type=int, default=1000)
    parser.add_argument("--communities", type=int, default=15)
    parser.add_argument("--r", type=float, default=0.1)
    parser.add_argument("--p-list", type=float, nargs="+", default=[0.4])
    parser.add_argument("--dim", type=int, default=2, help="Embedding dimension for DiSBM.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.dataset == "disbm":
        df = run_disbm(args)
        by = ["p"]
    else:
        df = run_manifold(args)
        by = ["embedding", "N"]
    if args.dataset == "disbm":
        suffix = f"_dim{args.dim}"
    elif args.graph == "knn":
        suffix = f"_knn{args.k}_scale{args.eps_scale}_beta{args.beta}"
    else:
        suffix = (f"_radius{args.radius}_beta{args.beta}" if args.radius is not None
                  else f"_radius_eps0{args.eps0}_cut{args.cut}_beta{args.beta}")
    path = args.out / f"{args.dataset}{suffix}.csv"
    df.to_csv(path, index=False)
    print(f"\nSaved {len(df)} runs to {path}\nMedian ± IQR over seeds:")
    print(summarize(df, by).to_string())


if __name__ == "__main__":
    main()
