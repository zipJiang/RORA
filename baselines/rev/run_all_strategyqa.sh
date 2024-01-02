for rationale_format in g gl s l
do 
    if ! test -d "/scratch/ylu130/project/REV_reimpl/ckpt/t5-strategyqa_${rationale_format}_1_none_mask"; then
        python steps/train_rev_model.py \
                --task-name t5-strategyqa \
                --rationale-format $rationale_format
    fi
    
    echo "Evaluating rationale format: $rationale_format" >> baselines/rev/rev.txt
    python steps/eval_rev_with_model.py \
            --dataset-dir data/processed_datasets/strategyqa \
            --model-dir /scratch/ylu130/project/REV_reimpl/ckpt/t5-strategyqa_${rationale_format}_1_none_mask \
            --rationale-format $rationale_format >> baselines/rev/rev.txt
done