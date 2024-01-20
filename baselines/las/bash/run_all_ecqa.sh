for rationale_format in ground_truth s l gl  
do 
    python baselines/las/run_tasks.py \
            --gpu 0 \
            -e ECQA.SIM.human \
            -b 32 \
            -g 3 \
            --explanations_to_use $rationale_format \
            --save_dir /scratch/ylu130/project/REV_reimpl/baseline/las/saved_models \
            --cache_dir /scratch/ylu130/project/REV_reimpl/baseline/las/cached_models
    
    python baselines/las/compute_sim.py \
            --model_name sim.human \
            --explanations_to_use $rationale_format \
            --gpu 0 \
            --split_name test \
            --data ECQA \
            --seed 21 \
            --bootstrap \
            --base_dir /scratch/ylu130/project/REV_reimpl/baseline/las >> baselines/las/training_reports/las_ecqa.txt
done

for data_name in gpt-4_demo=2_raw=True gpt-3.5-turbo_demo=2_raw=True gpt2_demo=0_raw=False t5-large_demo=0_raw=False
do 
    echo "Evaluating on: ${data_name}" >> baselines/las/training_reports/las_ecqa_model_rationale.txt
    python baselines/las/compute_sim.py \
            --model_name sim.human \
            --explanations_to_use ground_truth \
            --gpu 0 \
            --split_name test \
            --data ECQAModel \
            --model_generated_rationale_name ${data_name} \
            --seed 21 \
            --bootstrap \
            --base_dir /scratch/ylu130/project/REV_reimpl/baseline/las >> baselines/las/training_reports/las_ecqa_model_rationale.txt
done