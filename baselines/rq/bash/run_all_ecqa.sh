
for rationale_format in g gl s l
do 
    # training (IR --> O) and evaluation
    python baselines/rq/rationale_to_label.py \
            --output_dir /scratch/ylu130/project/REV_reimpl/baseline/rq/ir2o \
            --task_name ecqa \
            --do_train \
            --num_train_epochs 200 \
            --per_device_train_batch_size 32 \
            --per_device_eval_batch_size 32 \
            --logging_first_step \
            --logging_steps 1 \
            --save_steps 1 \
            --save_total_limit 11 \
            --seed 42 \
            --early_stopping_threshold 10 \
            --use_dev_real_expls \
            --include_input \
            --do_eval \
            --test_predict \
            --rationale_format $rationale_format
done