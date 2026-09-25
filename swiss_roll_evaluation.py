import torch
from finsler_embedding.eval_operators import *


def main():
    beta_list = torch.linspace(0.01, 0.9, 10)
    N_list = [500, 1000, 2000, 4000]
    df_list = []
    for beta in [0.01, 0.1, 0.2, 0.5, 0.8]:
        for N in [100, 500, 1000, 2000]:
            config: EvalOperators = EvalOperators(
                dataset="swiss_roll",
                n=N,
                m=2,
                beta=beta,
                graph_type="radius",
                embedding_type="isomap",
                k=15,
                eps_scale=1.0,
                eps0=1.0,
                cut=0.0,
                radius=5.0,
            )
        df = main(config)
        df_list.append(df)
    df_tot = pd.concat(df_list, ignore_index=True)

    df_tot.to_csv("./results/eval_operators_swiss_roll.csv", index=False)
