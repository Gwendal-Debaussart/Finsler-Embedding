"""Run quantitative evaluations on the sphere dataset."""

import argparse

from finsler_embedding.experiments.legacy.evaluation import run_manifold

from .config import SphereConfig


def parse_args() -> SphereConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=SphereConfig.n)
    parser.add_argument("--beta", type=float, default=SphereConfig.beta)
    parser.add_argument("--seeds", type=int, default=SphereConfig.seeds)
    parser.add_argument("--graph", choices=["knn", "radius"], default=SphereConfig.graph)
    parser.add_argument("--k", type=int, default=SphereConfig.k)
    parser.add_argument("--eps-scale", type=float, default=SphereConfig.eps_scale)
    parser.add_argument("--eps0", type=float, default=SphereConfig.eps0)
    parser.add_argument("--cut", type=float, default=SphereConfig.cut)
    parser.add_argument("--radius", type=float, default=SphereConfig.radius)
    parser.add_argument("--device", default=SphereConfig.device)
    parser.add_argument("--embedding", nargs="+", choices=["chart", "isomap"],
                        default=list(SphereConfig.embeddings))
    parser.add_argument("--out", default=SphereConfig.output)
    args = parser.parse_args()
    return SphereConfig(
        n=args.n, beta=args.beta, seeds=args.seeds, graph=args.graph, k=args.k,
        eps_scale=args.eps_scale, eps0=args.eps0, cut=args.cut, radius=args.radius,
        device=args.device, embeddings=tuple(args.embedding), output=args.out,
    )


def run(config: SphereConfig):
    """Run one sphere sweep and return its metrics."""
    namespace = argparse.Namespace(
        dataset="sphere",
        n=config.n,
        n_list=None,
        beta=config.beta,
        seeds=config.seeds,
        graph=config.graph,
        k=config.k,
        eps_scale=config.eps_scale,
        eps0=config.eps0,
        cut=config.cut,
        radius=config.radius,
        device=config.device,
        embedding=list(config.embeddings),
    )
    return run_manifold(namespace)


def main() -> None:
    config = parse_args()
    config.output.mkdir(parents=True, exist_ok=True)
    result = run(config)
    path = config.output / "sphere.csv"
    result.to_csv(path, index=False)
    print(f"Saved {len(result)} runs to {path}")


if __name__ == "__main__":
    main()
