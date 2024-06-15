for data_name in gpt-4_demo=2_raw=True gpt-3.5-turbo_demo=2_raw=True gpt2_demo=0_raw=False t5-large_demo=0_raw=False
do 
    # evaluate (IR --> O) on model-generated rationales
    python baselines/rq/rationale_to_label.py \
            --output_dir /scratch/ylu130/project/REV_reimpl/baseline/rq/ir2o \
            --task_name strategyqa_model \
            --pretrained_model_file /scratch/ylu130/project/REV_reimpl/baseline/rq/ir2o/010224_232038_g \
            --model_generated_rationale_name $data_name \
            --use_dev_real_expls \
            --per_device_eval_batch_size 64 \
            --seed 42 \
            --test_predict \
            --include_input \
            --rationale_format g
done
