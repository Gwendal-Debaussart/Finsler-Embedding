cd /home/tblanchard/phd/Finsler-Embedding
base_export_path="/home/tblanchard/phd/Finsler-Embedding/results/ablation_neighbours"

n_list=(5 10 15 20 25 30 50 100 -1)
for n in "${n_list[@]}"; do
    export_path="$base_export_path/n_${n}"
    uv run src/finsler_embedding/experiment_run.py \
        --export_path $export_path \
        --N 5000 \
        --K 400 \
        --omega_type constant \
        --no_plot
done

