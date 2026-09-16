cd /home/tblanchard/phd/Finsler-Embedding
base_export_path="/home/tblanchard/phd/Finsler-Embedding/results/ablation_n_pts"

n_list=(100 200 400 800 1600 3200 6400 12800 25600)
K_list=(200 1600 )
for n in "${n_list[@]}"; do
    for K in "${K_list[@]}"; do
        export_path="$base_export_path/n_${n}_K_${K}"
        uv run src/finsler_embedding/experiment_run.py \
        --export_path $export_path \
        --N $n \
        --K $K \
        --omega_type round \
        --function_type mexican \
        --no_plot
    done
done

