for data_name in gpt-4_demo=2_raw=True gpt-3.5-turbo_demo=2_raw=True gpt2_demo=0_raw=False t5-large_demo=0_raw=False
do 
    echo "Evaluating on: ${data_name}" >> baselines/las/training_reports/las_model_rationale.txt
    python baselines/las/compute_sim.py \
            --model_name sim.human \
            --explanations_to_use ground_truth \
            --gpu 0 \
            --split_name test \
            --data StrategyQAModel \
            --model_generated_rationale_name ${data_name} \
            --seed 21 \
            --bootstrap \
            --base_dir /scratch/ylu130/project/REV_reimpl/baseline/las >> baselines/las/training_reports/las_model_rationale.txt
done