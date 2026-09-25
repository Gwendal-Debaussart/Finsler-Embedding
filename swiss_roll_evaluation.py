import torch
import pandas as pd

from finsler_embedding.eval_operators import main as main_eval
from finsler_embedding.eval_operators import EvalOperators


def main():
    beta_list = torch.linspace(0.01, 0.9, 10)
    N_list = [1000, 2000, 4000]
    df_list = []
    for beta in beta_list:
        for N in N_list:
            config: EvalOperators = EvalOperators(
                dataset="swiss_roll",
                n=N,
                m=2,
                beta=beta.item(),
                graph_type="radius",
                embedding_type="isomap",
                k=15,
                eps_scale=1.0,
                eps0=1.0,
                cut=0.0,
                radius=5.0, # <- pas sur du tout de la valeur ici 
            )
            df = main_eval(config)
            df_list.append(df)
            pd.concat(df_list, ignore_index=True).to_csv(
                "./results/eval_operators_swiss_roll.csv", index=False
            )

    df_tot = pd.concat(df_list, ignore_index=True)
    df_tot.to_csv("./results/eval_operators_swiss_roll.csv", index=False)


if __name__ == "__main__":
    main()
