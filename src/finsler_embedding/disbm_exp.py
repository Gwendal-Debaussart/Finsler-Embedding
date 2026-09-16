import torch
import matplotlib.pyplot as plt
import networkx as nx
from argparse import ArgumentParser
from dataclasses import dataclass
from sklearn.manifold import Isomap, SpectralEmbedding
from finsler_embedding.experiment_run import *
from finsler_embedding.disbm import *


@dataclass
class DSBMExperimentConfig:
    N: int = 2000
    r: float = 0.1
    p: float = 0.4
    K: int = 15
    eps: float = 1.0
    m: int = 2
    export_path: Path = Path("./results/disbm_experiment")
    device = "cpu"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--N", type=int, default=2000, help="Number of nodes in the graph")
    parser.add_argument("--r", type=float, default=0.1, help="Self community edge probability")
    parser.add_argument(
        "--p", type=float, default=0.4, help="Forward community edge probability"
    )
    parser.add_argument("--K", type=int, default=15, help="Number of communities")
    parser.add_argument("--eps", type=float, default=1.0, help="Epsilon for kernel")
    parser.add_argument("--m", type=int, default=2, help="Dimension of the embedding")
    parser.add_argument(
        "--export_path",
        type=str,
        default="./results/disbm_experiment",
        help="Path to export results",
    )

    args = parser.parse_args()
    config = DSBMExperimentConfig(
        N=args.N,
        r=args.r,
        p=args.p,
        K=args.K,
        eps=args.eps,
        m=args.m,
        export_path=Path(args.export_path).resolve(),
    )
    return config


def main(cfg: DSBMExperimentConfig):
    cfg.export_path.mkdir(parents=True, exist_ok=True)

    q = 1 - cfg.p - cfg.r

    edges, dst_edges, community_labels, A = graph_info(
        p=cfg.p,
        q=q,
        r=cfg.r,
        K=cfg.K,
        N=cfg.N,
    )

    W, mu_0, mu_1 = gaussian_kernel(edges, dst_edges, cfg.eps, cfg.N, cfg.m)
    c_km = -mu_1 / mu_0 * (cfg.m + 1) / cfg.m

    Q_inv_theta = get_Q_inv_theta(W, theta=1)
    W_theta_s, W_theta_a = get_W_thetas(W, Q_inv_theta)
    P_theta_s, P_theta_a = construct_P_thetas(W_theta_s, W_theta_a)
    L_theta_s = 1 / cfg.eps**2 * P_theta_s
    L_theta_a = 1 / cfg.eps * P_theta_a

    X_low = SpectralEmbedding(n_components=2, affinity='precomputed').fit_transform(
        W_theta_s.detach().cpu().numpy()
    )
    X_low = torch.from_numpy(X_low).float().to(cfg.device)

    base_cometric = IdentityCoMetric()
    # base_cometric = CentroidsCometric(centroids=X_low,cometric_centroids=base_cometric(X_low))

    f_values_low, Lf_values_low, f_grad_values_low = prepare_test_functions(
        # cfg.K, "mexican", X, L_theta_a
        cfg.K,
        "coordinates",
        X_low,
        L_theta_a,
    )

    v_model, omega_model, loss_list = train_vector_field_models(
        X_low,
        base_cometric,
        c_km,
        f_values_low,
        Lf_values_low,
        f_grad_values_low,
        m=cfg.m,
        n_epochs=1000,
    )
    v_hat_deep_low = v_model(X_low).detach()
    b_hat_deep_low = omega_model(X_low).detach()

    graph = nx.from_numpy_array(A.numpy())

    skip = 10
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].scatter(
        X_low[:, 0].detach().cpu(),
        X_low[:, 1].detach().cpu(),
        c=community_labels,
        cmap="viridis",
        s=5,
    )
    axes[0].set_title("Eigenvectors of Symmetric Operator")
    axes[1].scatter(
        X_low[:, 0].detach().cpu(),
        X_low[:, 1].detach().cpu(),
        c=community_labels,
        cmap="viridis",
        s=5,
    )
    axes[1].quiver(
        X_low[::skip, 0].detach().cpu(),
        X_low[::skip, 1].detach().cpu(),
        b_hat_deep_low[::skip, 0].detach().cpu(),
        b_hat_deep_low[::skip, 1].detach().cpu(),
        color="red",
        scale=0.2,
        alpha=0.5,
        angles="xy",
        scale_units="xy",
    )
    axes[1].set_title("Learned Randers Vector Field on Embedding")
    nx.draw(
        graph,
        with_labels=False,
        node_color=community_labels.numpy(),
        cmap=plt.cm.viridis,
        ax=axes[2],
    )
    for ax in axes:
        ax.axis("off")
    plt.savefig(cfg.export_path / "disbm_sidebyside.pdf", dpi=300)
    plt.close(fig)

    # Meshgrid for vector field visualization
    bounds_grid = get_bounds(X_low, margin=0.4)
    bounds_plts = get_bounds(X_low, margin=0.1)
    x = torch.linspace(bounds_grid[0], bounds_grid[1], 20)
    y = torch.linspace(bounds_grid[2], bounds_grid[3], 20)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    grid_points = torch.stack([X.flatten(), Y.flatten()], dim=1)
    omega_grid = omega_model(grid_points).detach()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        X_low[:, 0].detach().cpu(),
        X_low[:, 1].detach().cpu(),
        c=community_labels,
        cmap="tab10",
        s=5,
    )
    ax.quiver(
        grid_points[:, 0].detach().cpu(),
        grid_points[:, 1].detach().cpu(),
        omega_grid[:, 0].detach().cpu(),
        omega_grid[:, 1].detach().cpu(),
        color="black",
        scale=0.2,
        alpha=0.5,
        angles="xy",
        scale_units="xy",
    )
    ax.axis("off")
    ax.set_xlim(bounds_plts[0], bounds_plts[1])
    ax.set_ylim(bounds_plts[2], bounds_plts[3])
    plt.savefig(cfg.export_path / "disbm_vector_field.pdf", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    cfg = parse_args()
    main(cfg)
