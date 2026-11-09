"""Run quantitative evaluations on the Matsumoto dataset."""

import argparse

from finsler_embedding.experiments.legacy.evaluation import run_manifold

from .config import MatsumotoConfig


def parse_args() -> MatsumotoConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=MatsumotoConfig.n)
    parser.add_argument("--seeds", type=int, default=MatsumotoConfig.seeds)
    parser.add_argument("--graph", choices=["knn", "radius"], default=MatsumotoConfig.graph)
    parser.add_argument("--k", type=int, default=MatsumotoConfig.k)
    parser.add_argument("--eps-scale", type=float, default=MatsumotoConfig.eps_scale)
    parser.add_argument("--eps0", type=float, default=MatsumotoConfig.eps0)
    parser.add_argument("--cut", type=float, default=MatsumotoConfig.cut)
    parser.add_argument("--radius", type=float, default=MatsumotoConfig.radius)
    parser.add_argument("--device", default=MatsumotoConfig.device)
    parser.add_argument("--embedding", nargs="+", choices=["chart", "isomap"],
                        default=list(MatsumotoConfig.embeddings))
    parser.add_argument("--out", default=MatsumotoConfig.output)
    args = parser.parse_args()
    return MatsumotoConfig(
        n=args.n, seeds=args.seeds, graph=args.graph, k=args.k,
        eps_scale=args.eps_scale, eps0=args.eps0, cut=args.cut, radius=args.radius,
        device=args.device, embeddings=tuple(args.embedding), output=args.out,
    )


def run(config: MatsumotoConfig):
    """Run one Matsumoto sweep and return its metrics."""
    namespace = argparse.Namespace(
        dataset="matsumoto", n=config.n, n_list=None, beta=None,
        seeds=config.seeds, graph=config.graph, k=config.k,
        eps_scale=config.eps_scale, eps0=config.eps0, cut=config.cut,
        radius=config.radius, device=config.device,
        embedding=list(config.embeddings),
    )
    return run_manifold(namespace)


def main() -> None:
    config = parse_args()
    config.output.mkdir(parents=True, exist_ok=True)
    result = run(config)
    path = config.output / "matsumoto.csv"
    result.to_csv(path, index=False)
    print(f"Saved {len(result)} runs to {path}")


if __name__ == "__main__":
    main()
