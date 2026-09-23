
start_time=$(date +%s)

cd /home/tblanchard/phd/Finsler-Embedding
base_export_path="/home/tblanchard/phd/Finsler-Embedding/results/ablation_epsilon"

N=10000
K=500
epsilon_list=(0.001 0.003 0.005 0.01 0.02 0.03 0.05 0.1 0.2 0.5 1.0)
for epsilon in "${epsilon_list[@]}"; do
        export_path="$base_export_path/epsilon_${epsilon}"
        uv run src/finsler_embedding/experiment_run.py \
        --export_path $export_path \
        --N $N \
        --K $K \
        --omega_type round \
        --function_type mexican \
        --no_plot \
        --epsilon $epsilon
done

end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
hours=$((elapsed_time / 3600))
minutes=$(((elapsed_time % 3600) / 60))
seconds=$((elapsed_time % 60))
echo "Total elapsed time: ${hours}h${minutes}min${seconds}s"