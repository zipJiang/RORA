for rationale_format in g gl s l
do
    for irm_coef in 100.0 10.0
    do  
        # train generator
        if ! test -d "/scratch/ylu130/project/REV_reimpl/ckpt/generation/strategyqa_t5-base_${rationale_format}_0.1"; then
            echo "Training generator with rationale format: $rationale_format"
            python steps/train_generator.py \
                    --rationale-format ${rationale_format} \
                    --removal-threshold 0.1
        fi

        # train irm model  
        if ! test -d "/scratch/ylu130/project/REV_reimpl/ckpt/irm/strategyqa_t5-base_${rationale_format}_0.1_${irm_coef}"; then
            echo "Training IRM evaluation model with rationale format $rationale_format and IRM coefficient $irm_coef"
            python steps/train_irm_model.py \
                    --rationale-format ${rationale_format} \
                    --removal-threshold 0.1 \
                    --irm-coefficient ${irm_coef}
        fi

        # evaluate on synthetic rationale test set
        echo "Evaluating rationale format: $rationale_format with IRM coefficient $irm_coef" >> log/rev2.txt
        python steps/eval_rev_with_model.py \
                --dataset-dir data/processed_datasets/strategyqa \
                --model-dir /scratch/ylu130/project/REV_reimpl/ckpt/irm/strategyqa_t5-base_${rationale_format}_0.1_${irm_coef} \
                --rationale-format ${rationale_format} >> log/rev2.txt
    done
done    

# # evaluate on model generated rationales
# for data_name in gpt-3.5-turbo_demo=2_raw=True gpt-4_demo=2_raw=True gpt2_demo=0_raw=False t5-large_demo=0_raw=False
# do 
#     for irm_coef in 100.0 10.0
#     do
#     echo "Evaluating model-generated ratinoales: $data_name with IRM coefficient $irm_coef" >> log/rev2.txt
#     python steps/eval_rev_with_model.py \
#             --dataset-dir data/processed_datasets/strategyqa_model_rationale/${data_name} \
#             --model-dir /scratch/ylu130/project/REV_reimpl/irm/strategyqa_t5-base_g_0.1_${irm_coef} \
#             --rationale-format g >> log/rev2.txt
#     done
# done