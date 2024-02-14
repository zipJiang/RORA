### ------ Experiments for synthetic rationales on REV2 ------ ###
# for remove_threshold in 0.001 0.005 0.01 0.05 0.1
# do
#     echo "Experiments for synthetic rationales on REV2 (t5-base, removal threshold=${remove_threshold})" >> log/rev2_t5.txt
#     for rationale_format in g gl s l
#     do
#         for irm_coef in 100.0 10.0
#         do  
#             # train generator
#             if ! test -d "/scratch/ylu130/project/REV_reimpl/ckpt/generation/strategyqa_t5-base_${rationale_format}_${remove_threshold}"; then
#                 echo "Training generator with rationale format: $rationale_format"
#                 python steps/train_generator.py \
#                         --rationale-format ${rationale_format} \
#                         --removal-threshold ${remove_threshold}
#             fi

#             # train irm model  
#             if ! test -d "/scratch/ylu130/project/REV_reimpl/ckpt/irm/strategyqa_t5-base_${rationale_format}_${remove_threshold}_${irm_coef}"; then
#                 echo "Training IRM evaluation model with rationale format $rationale_format and IRM coefficient $irm_coef"
#                 python steps/train_irm_model.py \
#                         --rationale-format ${rationale_format} \
#                         --removal-threshold ${remove_threshold} \
#                         --irm-coefficient ${irm_coef}
#             fi

#             # evaluate on synthetic rationale test set
#             echo "Evaluating rationale format: $rationale_format with IRM coefficient $irm_coef" >> log/rev2_t5.txt
#             python steps/eval_rev_with_model.py \
#                     --dataset-dir data/processed_datasets/strategyqa \
#                     --model-dir /scratch/ylu130/project/REV_reimpl/ckpt/irm/strategyqa_t5-base_${rationale_format}_${remove_threshold}_${irm_coef} \
#                     --rationale-format ${rationale_format} >> log/rev2_t5.txt
#         done
#     done    
# done

### ------ Experiments for model generated rationales on REV2 ------ ###
# for data_name in gpt-4_demo=2_raw=True gpt-3.5-turbo_demo=2_raw=True Llama-2-7b-hf_demo=2_raw=True flan-t5-large_demo=2_raw=True
# do 
#     for split in train validation test
#     do
#         python steps/rationale_preprocessing.py \
#                 --data-handle Yining/generated_rationales/strategyqa \
#                 --split $split \
#                 --write-to data/processed_datasets/strategyqa_model_rationale \
#                 --data-name $data_name
#     done
# done

# for remove_threshold in 0.001 0.005 0.01 0.05 0.1
# do 
#     echo "Experiments for model generated rationales on REV2 (t5-base, removal threshold=${remove_threshold})" >> log/rev2_t5.txt
#     for model in gpt-4_demo=2_raw=True gpt-3.5-turbo_demo=2_raw=True Llama-2-7b-hf_demo=2_raw=True flan-t5-large_demo=2_raw=True
#     do 
#         for irm_coef in 100.0 10.0
#         do  
#             # train generator
#             if ! test -d "/scratch/ylu130/project/REV_reimpl/ckpt/generation/strategyqa_model_rationale_t5-base_g_${remove_threshold}_${model}"; then
#                 echo "Training generator with data name: $model"
#                 python steps/train_generator.py \
#                         --task-name strategyqa_model_rationale \
#                         --rationale-format g \
#                         --removal-threshold ${remove_threshold} \
#                         --data-name $model
#             fi
#             # train irm model  
#             if [ ! -d "/scratch/ylu130/project/REV_reimpl/ckpt/irm/strategyqa_model_rationale_t5-base_g_${remove_threshold}_${irm_coef}_${model}" ]; then
#                 echo "Training IRM evaluation model with data $model and IRM coefficient $irm_coef"
#                 python steps/train_irm_model.py \
#                         --task-name strategyqa_model_rationale \
#                         --rationale-format g \
#                         --removal-threshold $remove_threshold \
#                         --irm-coefficient $irm_coef \
#                         --data-name $model
#             fi

#             # evaluate on synthetic rationale test set
#             echo "Evaluating rationale format: $model with IRM coefficient $irm_coef" >> log/rev2_t5.txt
#             python steps/eval_rev_with_model.py \
#                     --dataset-dir data/processed_datasets/strategyqa_model_rationale/${model} \
#                     --model-dir /scratch/ylu130/project/REV_reimpl/ckpt/irm/strategyqa_model_rationale_t5-base_g_${remove_threshold}_${irm_coef}_${model} \
#                     --rationale-format g >> log/rev2_t5.txt
#         done
#     done    
# done

# ## ------ Experiments for synthetic rationales on REV2 (deberta-v3-large) ------ ###
# for remove_threshold in 0.001 0.005 0.01 0.05 0.1
# do 
#     echo "Experiments for synthetic rationales on REV2 (deberta-v3-large, removal threshold=${remove_threshold})" >> log/rev2.txt
#     for rationale_format in g gl s l
#     do
#         for irm_coef in 100.0 10.0
#         do  
#             # train generator
#             if ! test -d "/scratch/ylu130/project/REV_reimpl/ckpt/generation/strategyqa_t5-base_${rationale_format}_${remove_threshold}"; then
#                 echo "Training generator with rationale format: $rationale_format"
#                 python steps/train_generator.py \
#                         --task-name strategyqa \
#                         --rationale-format ${rationale_format} \
#                         --removal-threshold ${remove_threshold}
#             fi

#             # train irm model  
#             if [ ! -d "/scratch/ylu130/project/REV_reimpl/ckpt/irm/strategyqa_microsoft::deberta-v3-large_${rationale_format}_${remove_threshold}_${irm_coef}" ]; then
#                 echo "Training IRM evaluation model with rationale format $rationale_format and IRM coefficient $irm_coef"
#                 python steps/train_irm_model.py \
#                         --rationale-format ${rationale_format} \
#                         --removal-threshold ${remove_threshold} \
#                         --irm-coefficient ${irm_coef} \
#                         --model-name microsoft/deberta-v3-large
#             fi

#             # evaluate on synthetic rationale test set
#             echo "Evaluating rationale format: $rationale_format with IRM coefficient $irm_coef" >> log/rev2.txt
#             python steps/eval_rev_with_model.py \
#                     --dataset-dir data/processed_datasets/strategyqa \
#                     --model-dir /scratch/ylu130/project/REV_reimpl/ckpt/irm/strategyqa_microsoft::deberta-v3-large_${rationale_format}_${remove_threshold}_${irm_coef} \
#                     --rationale-format ${rationale_format} >> log/rev2.txt
#         done
#     done
# done

# if [ ! -d "/scratch/ylu130/project/REV_reimpl/ckpt/deberta-strategyqa_n_1_none_mask" ]; then
#     python steps/train_rev_model.py \
#             --task-name deberta-strategyqa \
#             --rationale-format n
# fi

# echo "Evaluating rationale format: n" >> log/rev2.txt
# python steps/eval_rev_with_model.py \
#         --dataset-dir data/processed_datasets/strategyqa \
#         --model-dir /scratch/ylu130/project/REV_reimpl/ckpt/deberta-strategyqa_n_1_none_mask \
#         --rationale-format n >> log/rev2.txt

# ### ------ Experiments for model generated rationales on REV2 (deberta-v3-large) ------ ###
# for remove_threshold in 0.001 0.005 0.01 0.05 0.1
# do 
#     echo "Experiments for model generated rationales on REV2 (deberta-v3-large, removal threshold=${remove_threshold})" >> log/rev2_model.txt
#     for model in gpt-4_demo=2_raw=True gpt-3.5-turbo_demo=2_raw=True Llama-2-7b-hf_demo=2_raw=True flan-t5-large_demo=2_raw=True
#     do 
#         for irm_coef in 100.0 10.0
#         do  
#             # train generator
#             if ! test -d "/scratch/ylu130/project/REV_reimpl/ckpt/generation/strategyqa_model_rationale_t5-base_g_${remove_threshold}_${model}"; then
#                 echo "Training generator with data name: $model"
#                 python steps/train_generator.py \
#                         --task-name strategyqa_model_rationale \
#                         --rationale-format g \
#                         --removal-threshold ${remove_threshold} \
#                         --data-name $model
#             fi
#             # train irm model  
#             if [ ! -d "/scratch/ylu130/project/REV_reimpl/ckpt/irm/strategyqa_model_rationale_microsoft::deberta-v3-large_g_${remove_threshold}_${irm_coef}_${model}" ]; then
#                 echo "Training IRM evaluation model with data $model and IRM coefficient $irm_coef"
#                 python steps/train_irm_model.py \
#                         --task-name strategyqa_model_rationale \
#                         --rationale-format g \
#                         --removal-threshold $remove_threshold \
#                         --irm-coefficient $irm_coef \
#                         --data-name $model \
#                         --model-name microsoft/deberta-v3-large \
#                         --batch-size 4
#             fi

#             # evaluate on synthetic rationale test set
#             echo "Evaluating rationale format: $model with IRM coefficient $irm_coef" >> log/rev2_model.txt
#             python steps/eval_rev_with_model.py \
#                     --dataset-dir data/processed_datasets/strategyqa_model_rationale/${model} \
#                     --model-dir /scratch/ylu130/project/REV_reimpl/ckpt/irm/strategyqa_model_rationale_microsoft::deberta-v3-large_g_${remove_threshold}_${irm_coef}_${model} \
#                     --rationale-format g >> log/rev2_model.txt
#         done
#     done    
# done

## ------ Sensitivity Test for IRM Coefficient ------ ##
## ------ Experiments for synthetic rationales on REV2 (deberta-v3-large) ------ ###
# for remove_threshold in 0.005
# do 
#     echo "Experiments for synthetic rationales on REV2 (deberta-v3-large, removal threshold=${remove_threshold})" >> log/rev2_deberta_lambda.txt
#     for rationale_format in g gl s l
#     do
#         for irm_coef in 0.0 1.0 5.0 10.0 50.0 100.0 500.0 1000.0
#         do  
#             # train generator
#             if ! test -d "/scratch/ylu130/project/REV_reimpl/ckpt/generation/strategyqa_t5-base_${rationale_format}_${remove_threshold}"; then
#                 echo "Training generator with rationale format: $rationale_format"
#                 python steps/train_generator.py \
#                         --task-name strategyqa \
#                         --rationale-format ${rationale_format} \
#                         --removal-threshold ${remove_threshold}
#             fi

#             # train irm model  
#             if [ ! -d "/scratch/ylu130/project/REV_reimpl/ckpt/irm/strategyqa_microsoft::deberta-v3-large_${rationale_format}_${remove_threshold}_${irm_coef}" ]; then
#                 echo "Training IRM evaluation model with rationale format $rationale_format and IRM coefficient $irm_coef"
#                 python steps/train_irm_model.py \
#                         --rationale-format ${rationale_format} \
#                         --removal-threshold ${remove_threshold} \
#                         --irm-coefficient ${irm_coef} \
#                         --model-name microsoft/deberta-v3-large
#             fi

#             # evaluate on synthetic rationale test set
#             echo "Evaluating rationale format: $rationale_format with IRM coefficient $irm_coef" >> log/rev2_deberta_lambda.txt
#             python steps/eval_rev_with_model.py \
#                     --dataset-dir data/processed_datasets/strategyqa \
#                     --model-dir /scratch/ylu130/project/REV_reimpl/ckpt/irm/strategyqa_microsoft::deberta-v3-large_${rationale_format}_${remove_threshold}_${irm_coef} \
#                     --rationale-format ${rationale_format} >> log/rev2_deberta_lambda.txt
#         done
#     done
# done

# if [ ! -d "/scratch/ylu130/project/REV_reimpl/ckpt/deberta-strategyqa_n_1_none_mask" ]; then
#     python steps/train_rev_model.py \
#             --task-name deberta-strategyqa \
#             --rationale-format n
# fi

# echo "Evaluating rationale format: n" >> log/rev2_deberta_lambda.txt
# python steps/eval_rev_with_model.py \
#         --dataset-dir data/processed_datasets/strategyqa \
#         --model-dir /scratch/ylu130/project/REV_reimpl/ckpt/deberta-strategyqa_n_1_none_mask \
#         --rationale-format n >> log/rev2_deberta_lambda.txt

## ------ Sensitivity Test for IRM Coefficient ------ ##
### ------ Experiments for model generated rationales on REV2 (deberta-v3-large) ------ ###
for remove_threshold in 0.005
do 
    echo "Experiments for model generated rationales on REV2 (deberta-v3-large, removal threshold=${remove_threshold})" >> log/rev2_deberta_lambda_model.txt
    for model in gpt-4_demo=2_raw=True gpt-3.5-turbo_demo=2_raw=True Llama-2-7b-hf_demo=2_raw=True flan-t5-large_demo=2_raw=True
    do 
        for irm_coef in 0.0 1.0 5.0 10.0 50.0 100.0 500.0 1000.0
        do  
            # train generator
            if ! test -d "/scratch/ylu130/project/REV_reimpl/ckpt/generation/strategyqa_model_rationale_t5-base_g_${remove_threshold}_${model}"; then
                echo "Training generator with data name: $model"
                python steps/train_generator.py \
                        --task-name strategyqa_model_rationale \
                        --rationale-format g \
                        --removal-threshold ${remove_threshold} \
                        --data-name $model
            fi
            # train irm model  
            if [ ! -d "/scratch/ylu130/project/REV_reimpl/ckpt/irm/strategyqa_model_rationale_microsoft::deberta-v3-large_g_${remove_threshold}_${irm_coef}_${model}" ]; then
                echo "Training IRM evaluation model with data $model and IRM coefficient $irm_coef"
                python steps/train_irm_model.py \
                        --task-name strategyqa_model_rationale \
                        --rationale-format g \
                        --removal-threshold $remove_threshold \
                        --irm-coefficient $irm_coef \
                        --data-name $model \
                        --model-name microsoft/deberta-v3-large \
                        --batch-size 4
            fi

            # evaluate on synthetic rationale test set
            echo "Evaluating rationale format: $model with IRM coefficient $irm_coef" >> log/rev2_deberta_lambda_model.txt
            python steps/eval_rev_with_model.py \
                    --dataset-dir data/processed_datasets/strategyqa_model_rationale/${model} \
                    --model-dir /scratch/ylu130/project/REV_reimpl/ckpt/irm/strategyqa_model_rationale_microsoft::deberta-v3-large_g_${remove_threshold}_${irm_coef}_${model} \
                    --rationale-format g >> log/rev2_deberta_lambda_model.txt
        done
    done    
done