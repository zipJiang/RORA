
## ------ Evaluation experiments for model-generated rationales on REV2 ------ ###
echo "Evaluation experiments for model-generated rationales on REV2" >> log/rev2_eval.txt
for data_name in gpt-4_demo=2_raw=True gpt-3.5-turbo_demo=2_raw=True Llama-2-7b-hf_demo=2_raw=True flan-t5-large_demo=2_raw=True t5-large_demo=0_raw=False gpt2_demo=0_raw=False
do 
    for irm_coef in 100.0 10.0
    do
        echo "Evaluating model-generated ratinoales: $data_name with IRM coefficient $irm_coef" >> log/rev2_eval.txt
        python steps/eval_rev_with_model.py \
                --dataset-dir data/processed_datasets/strategyqa_model_rationale/${data_name} \
                --model-dir /scratch/ylu130/project/REV_reimpl/ckpt/irm/strategyqa_t5-base_g_0.1_${irm_coef} \
                --rationale-format g >> log/rev2_eval.txt
    done
done


## ------ Evaluation experiments for synthetic rationales on REV2 ------ ###
echo "Evaluation experiments for synthetic rationales on REV2" >> log/rev2_eval.txt
for data_name in gl s l
do 
    for irm_coef in 100.0 10.0
    do
        echo "Evaluating synthetic ratinoales: $data_name with IRM coefficient $irm_coef" >> log/rev2_eval.txt
        python steps/eval_rev_with_model.py \
                --dataset-dir data/processed_datasets/strategyqa \
                --model-dir /scratch/ylu130/project/REV_reimpl/ckpt/irm/strategyqa_t5-base_g_0.1_${irm_coef} \
                --rationale-format $data_name >> log/rev2_eval.txt
    done
done
