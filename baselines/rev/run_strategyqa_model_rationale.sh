# for data_name in gpt-4_demo=2_raw=True gpt-3.5-turbo_demo=2_raw=True gpt2_demo=0_raw=False t5-large_demo=0_raw=False
# do 
#     echo "Evaluating model generated rationale: $data_name" >> log/rev.txt
#     python steps/eval_rev_with_model.py \
#             --dataset-dir data/processed_datasets/strategyqa_model_rationale/${data_name} \
#             --model-dir /scratch/ylu130/project/REV_reimpl/ckpt/t5-strategyqa_g_1_none_mask \
#             --rationale-format g >> log/rev.txt
# done

# for data_name in gpt-4_demo=2_raw=True gpt-3.5-turbo_demo=2_raw=True gpt2_demo=0_raw=False t5-large_demo=0_raw=False
# do 
#     echo "Evaluating model generated rationale: $data_name" using gs >> log/rev.txt
#     python steps/eval_rev_with_model.py \
#             --dataset-dir data/processed_datasets/strategyqa_model_rationale/${data_name} \
#             --model-dir /scratch/ylu130/project/REV_reimpl/ckpt/t5-strategyqa_gs_1_none_mask \
#             --rationale-format gs >> log/rev.txt
# done

for data_name in gpt-4_demo=2_raw=True gpt-3.5-turbo_demo=2_raw=True gpt2_demo=0_raw=False t5-large_demo=0_raw=False
do 
    echo "Evaluating model generated rationale: $data_name" using gs >> log/rev.txt
    python steps/eval_rev_with_model.py \
            --dataset-dir data/processed_datasets/strategyqa_model_rationale/${data_name} \
            --model-dir /scratch/ylu130/project/REV_reimpl/ckpt/t5-strategyqa_gs_1_none_mask_rationale_only \
            --rationale-format gs >> log/rev.txt
done