### ------ Experiments for synthetic rationales  ------ ###
# echo "Experiments for synthetic rationales" >> baselines/las/training_reports/las.txt
# for rationale_format in ground_truth s l gl  
# do 
#     python baselines/las/run_tasks.py \
#             --gpu 0 \
#             -e StrategyQA.SIM.human \
#             -b 32 \
#             -g 3 \
#             --explanations_to_use $rationale_format \
#             --save_dir /scratch/ylu130/project/REV_reimpl/baseline/las/saved_models \
#             --cache_dir /scratch/ylu130/project/REV_reimpl/baseline/las/cached_models
    
#     python baselines/las/compute_sim.py \
#             --model_name sim.human \
#             --explanations_to_use $rationale_format \
#             --gpu 0 \
#             --split_name test \
#             --data StrategyQA \
#             --seed 21 \
#             --bootstrap \
#             --base_dir /scratch/ylu130/project/REV_reimpl/baseline/las >> baselines/las/training_reports/las.txt
# done

### ------ Experiments for model-generated rationales  ------ ###
# echo "Experiments for model-generated rationales" >> baselines/las/training_reports/las.txt
for data_name in gpt-4_demo=2_raw=True gpt-3.5-turbo_demo=2_raw=True Llama-2-7b-hf_demo=2_raw=True flan-t5-large_demo=2_raw=True  
do 
    if ! test -e "/scratch/ylu130/project/REV_reimpl/baseline/las/saved_models/StrategyQAModel_t5-base_sim.human_seed21_rationale=ground_truth_data=${data_name}.hdf5"; then
        python baselines/las/run_tasks.py \
                --gpu 0 \
                -e StrategyQAModel.SIM.human \
                -b 32 \
                -g 3 \
                --explanations_to_use ground_truth \
                --save_dir /scratch/ylu130/project/REV_reimpl/baseline/las/saved_models \
                --cache_dir /scratch/ylu130/project/REV_reimpl/baseline/las/cached_models \
                --data_name $data_name
    fi
    
    echo "Evaluating on: ${data_name}" >> baselines/las/training_reports/las.txt
    python baselines/las/compute_sim.py \
            --model_name sim.human \
            --explanations_to_use ground_truth \
            --gpu 0 \
            --split_name test \
            --data StrategyQAModel \
            --model_generated_rationale_name ${data_name} \
            --data_name ${data_name} \
            --seed 21 \
            --bootstrap \
            --base_dir /scratch/ylu130/project/REV_reimpl/baseline/las >> baselines/las/training_reports/las.txt
done