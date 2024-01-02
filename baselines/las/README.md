## Prepare Simulator

*Setup Environment* This is important as the original environment is too old to be compatiable with the latest GPU cuda version

`pip install -r baselines/las/requirements.txt`

**Note** that for all modeling experiments below, `gpu`, `save_dir`, and `cache_dir` must be provided as args to argpase (recommended to make save_dir and cache_dir in same directory). `-b` and `-g` refer to train batch size and gradient accumulation factors, respectively (effective batch size is their product). 

*Human Simulator*: `python run_tasks.py --gpu gpu -e QA.SIM.human -b 4 -g 3 --save_dir save_dir --cache_dir cache_dir`

*StrategyQA Human Simulator*: `python baselines/las/run_tasks.py --gpu 0 -e StrategyQA.SIM.human -b 32 -g 3 --explanations_to_use ground_truth --save_dir /scratch/ylu130/project/REV_reimpl/baseline/las/saved_models --cache_dir /scratch/ylu130/project/REV_reimpl/baseline/las/cached_models`

## Computing LAS

We compute LAS scores with the `compute_sim.py` script. Here, `gpu` and `base_dir` must be provided as arguments. `base_dir` should include a `saved_models` and `cached_models` directories. For each condition, LAS computation is possible after running the respective experiments from above. Note `split_name` is `dev` for CQA to compare with human provided explanations, but should be `test` for SNLI.

*Human Simulator*: `python compute_sim.py --model_name sim.human --explanations_to_use ground_truth --gpu gpu --split_name dev --data QA --seed seed --bootstrap`

*StrategyQA Human Simulator*: `python baselines/las/compute_sim.py --model_name sim.human --explanations_to_use ground_truth --gpu 0 --split_name test --data StrategyQA --seed 21 --bootstrap --base_dir /scratch/ylu130/project/REV_reimpl/baseline/las`

## Run LAS on StrategyQA
**Run LAS on all rationale formats**: `bash baselines/las/bash/run_all_strategyqa.sh`

