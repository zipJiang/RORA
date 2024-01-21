## Measuring Association Between Labels and Free-Text Rationales

*Setup Environment*

`pip install -r baselines/rq/requirements.txt`

If you also encounter `TypeError: issubclass() arg 1 must be a class` when running `parser = HfArgumentParser(...)`, editing `__subclasscheck__` function (line 771) in `/home/ylu130/.conda/envs/rq/lib/python3.8/typing.py` as follows:

```
return issubclass(cls, self.__origin__) -> return issubclass(type(cls), self.__origin__)
```

## Run RQ on StrategyQA

Rationale quality of a set of rationales is computed as `IR-->O` performance minus `I-->O` performance

**Get I-->O baseline performance**:

```
python baselines/rq/input_to_label_and_rationale.py \
        --output_dir /scratch/ylu130/project/REV_reimpl/baseline/rq/i2o \
        --task_name strategyqa \
        --do_train \
        --num_train_epochs 200 \
        --per_device_train_batch_size 64 \
        --per_device_eval_batch_size 64 \
        --logging_first_step \
        --logging_steps 1 \
        --save_steps 1 \
        --save_total_limit 11 \
        --seed 42 \
        --early_stopping_threshold 10 \
        --do_eval \
        --test_predict \
        --label_only
```

**Run RQ on all rationale formats (IR-->O)**: `bash baselines/rq/bash/run_all_strategyqa.sh`

**Run RQ on all model generated rationales (IR-->O)**: `bash baselines/rq/bash/run_strategyqa_model_rationale.sh`

## Run RQ on ECQA

Rationale quality of a set of rationales is computed as `IR-->O` performance minus `I-->O` performance

**Get I-->O baseline performance**:

```
python baselines/rq/input_to_label_and_rationale.py \
        --output_dir /scratch/ylu130/project/REV_reimpl/baseline/rq/i2o \
        --task_name ecqa \
        --do_train \
        --num_train_epochs 200 \
        --per_device_train_batch_size 64 \
        --per_device_eval_batch_size 64 \
        --logging_first_step \
        --logging_steps 1 \
        --save_steps 1 \
        --save_total_limit 11 \
        --seed 42 \
        --early_stopping_threshold 10 \
        --do_eval \
        --test_predict \
        --label_only
```

**Run RQ on all rationale formats (IR-->O)**: `bash baselines/rq/bash/run_all_ecqa.sh`

**Run RQ on all model generated rationales (IR-->O)**: `bash baselines/rq/bash/run_ecqa_model_rationale.sh`

**Run RQ on COS-E**:

```
python baselines/rq/rationale_to_label.py \
        --output_dir /scratch/ylu130/project/REV_reimpl/baseline/rq/ir2o \
        --task_name cos_e \
        --version_name v1.11 \
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
        --dev_predict \
        --rationale_format g
```
