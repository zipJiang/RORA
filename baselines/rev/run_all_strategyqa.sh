# for rationale_format in g gl s l
# do 
#     if ! test -d "/scratch/ylu130/project/REV_reimpl/ckpt/t5-strategyqa_${rationale_format}_1_none_mask"; then
#         python steps/train_rev_model.py \
#                 --task-name t5-strategyqa \
#                 --rationale-format $rationale_format
#     fi
    
#     echo "Evaluating rationale format: $rationale_format" >> log/rev.txt
#     python steps/eval_rev_with_model.py \
#             --dataset-dir data/processed_datasets/strategyqa \
#             --model-dir /scratch/ylu130/project/REV_reimpl/ckpt/t5-strategyqa_${rationale_format}_1_none_mask \
#             --rationale-format $rationale_format >> log/rev.txt
# done


for rationale_format in gls gs ss ls s
do  
    # will not be used
    if ! test -e "data/processed_datasets/strategyqa/vocab_format=${rationale_format}_ng=2_mf=1_mt=10000_r=1.pt"; then
        python scripts/generate_vocabs.py \
                --dataset-dir data/processed_datasets/strategyqa \
                --rationale-format $rationale_format \
                --rationale-only
    fi
    # will not be used
    if ! test -d "/scratch/ylu130/project/REV_reimpl/ckpt/fasttext-strategyqa_${rationale_format}_1"; then
        python steps/train_rev_model.py \
                --task-name fasttext-strategyqa \
                --rationale-format $rationale_format 
    fi  


    if ! test -d "/scratch/ylu130/project/REV_reimpl/ckpt/t5-strategyqa_${rationale_format}_1_none_mask_rationale_only"; then
        python steps/train_rev_model.py \
                --task-name t5-strategyqa \
                --rationale-format $rationale_format \
                --rationale-only
    fi
    
    echo "Evaluating rationale format: $rationale_format" >> log/rev.txt
    python steps/eval_rev_with_model.py \
            --dataset-dir data/processed_datasets/strategyqa \
            --model-dir /scratch/ylu130/project/REV_reimpl/ckpt/t5-strategyqa_${rationale_format}_1_none_mask_rationale_only \
            --rationale-format $rationale_format >> log/rev.txt
done