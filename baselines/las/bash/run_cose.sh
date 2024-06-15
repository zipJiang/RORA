python baselines/las/run_tasks.py \
        --gpu 0 \
        -e COSE.SIM.human \
        -b 32 \
        -g 3 \
        --explanations_to_use ground_truth \
        --save_dir /scratch/ylu130/project/REV_reimpl/baseline/las/saved_models \
        --cache_dir /scratch/ylu130/project/REV_reimpl/baseline/las/cached_models

python baselines/las/compute_sim.py \
        --model_name sim.human \
        --explanations_to_use ground_truth \
        --gpu 0 \
        --split_name test \
        --data COSE \
        --seed 21 \
        --bootstrap \
        --base_dir /scratch/ylu130/project/REV_reimpl/baseline/las >> baselines/las/training_reports/las_cose.txt