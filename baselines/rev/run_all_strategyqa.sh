### ------ Experiments for synthetic rationales on REV (ablation study) ------ ###
echo "Experiments for synthetic rationales on REV (ablation study)" >> log/rev.txt
for rationale_format in g gl s l
do 
    if [ ! -d "/scratch/ylu130/project/REV_reimpl/ckpt/t5-strategyqa_${rationale_format}_1_none_mask" ]; then
        python steps/train_rev_model.py \
                --task-name t5-strategyqa \
                --rationale-format $rationale_format
    fi
    
    echo "Evaluating rationale format: $rationale_format" >> log/rev.txt
    python steps/eval_rev_with_model.py \
            --dataset-dir data/processed_datasets/strategyqa \
            --model-dir /scratch/ylu130/project/REV_reimpl/ckpt/t5-strategyqa_${rationale_format}_1_none_mask \
            --rationale-format $rationale_format >> log/rev.txt
done

### ------ Experiments for synthetic rationales on REV (ours) ------ ###
echo "Experiments for synthetic rationales on REV (ours)" >> log/rev.txt
for rationale_format in gls gs ss ls s
do  
    # will not be used
    if [ ! -e "data/processed_datasets/strategyqa/vocab_format=${rationale_format}_ng=2_mf=1_mt=10000_r=1.pt" ]; then
        python scripts/generate_vocabs.py \
                --dataset-dir data/processed_datasets/strategyqa \
                --rationale-format $rationale_format \
                --rationale-only
    fi
    # will not be used
    if [ ! -d "/scratch/ylu130/project/REV_reimpl/ckpt/fasttext-strategyqa_${rationale_format}_1" ]; then
        python steps/train_rev_model.py \
                --task-name fasttext-strategyqa \
                --rationale-format $rationale_format 
    fi  


    if [ ! -d "/scratch/ylu130/project/REV_reimpl/ckpt/t5-strategyqa_${rationale_format}_1_none_mask" ]; then
        python steps/train_rev_model.py \
                --task-name t5-strategyqa \
                --rationale-format $rationale_format
    fi
    
    echo "Evaluating rationale format: $rationale_format" >> log/rev.txt
    python steps/eval_rev_with_model.py \
            --dataset-dir data/processed_datasets/strategyqa \
            --model-dir /scratch/ylu130/project/REV_reimpl/ckpt/t5-strategyqa_${rationale_format}_1_none_mask \
            --rationale-format $rationale_format >> log/rev.txt
done

### ------ Experiments for synthetic rationales on REV (original) ------ ###
echo "Experiments for synthetic rationales on REV (original)" >> log/rev.txt
for rationale_format in gls gs ss ls s
do  
    # will not be used
    if [ ! -e "data/processed_datasets/strategyqa/vocab_format=${rationale_format}_ng=2_mf=1_mt=10000_r=1.pt" ]; then
        python scripts/generate_vocabs.py \
                --dataset-dir data/processed_datasets/strategyqa \
                --rationale-format $rationale_format \
                --rationale-only
    fi
    # will not be used
    if [ ! -d "/scratch/ylu130/project/REV_reimpl/ckpt/fasttext-strategyqa_${rationale_format}_1" ]; then
        python steps/train_rev_model.py \
                --task-name fasttext-strategyqa \
                --rationale-format $rationale_format 
    fi  


    if [ ! -d "/scratch/ylu130/project/REV_reimpl/ckpt/t5-strategyqa_${rationale_format}_1_none_mask_rationale_only" ]; then
        python steps/train_rev_model.py \
                --task-name t5-strategyqa \
                --rationale-format $rationale_format \
                --rationale-only
    fi
    
    echo "Evaluating rationale format: $rationale_format" >> log/rev.txt
    python steps/eval_rev_with_model.py \
            --dataset-dir data/processed_datasets/strategyqa \
            --model-dir /scratch/ylu130/project/REV_reimpl/ckpt/t5-strategyqa_${rationale_format}_1_none_mask_rationale_only \
            --rationale-format $rationale_format \
            --rationale-only   >> log/rev.txt
done

### ------ Experiments for model generated rationales on REV (ablation study) ------ ###
echo "Experiments for model generated rationales on REV (ablation study)" >> log/rev.txt
for data_name in gpt-4_demo=2_raw=True gpt-3.5-turbo_demo=2_raw=True Llama-2-7b-hf_demo=2_raw=True flan-t5-large_demo=2_raw=True
do 
    if [ ! -d "/scratch/ylu130/project/REV_reimpl/ckpt/t5-strategyqa_model_rationale_g_1_none_mask_${data_name}" ]; then
        python steps/train_rev_model.py \
                --task-name t5-strategyqa_model_rationale \
                --rationale-format g \
                --data-name $data_name
    fi
    
    echo "Evaluating data: $data_name" >> log/rev.txt
    python steps/eval_rev_with_model.py \
            --dataset-dir data/processed_datasets/strategyqa_model_rationale/${data_name} \
            --model-dir /scratch/ylu130/project/REV_reimpl/ckpt/t5-strategyqa_model_rationale_g_1_none_mask_${data_name} \
            --rationale-format g >> log/rev.txt
done 

### ------ Experiments for model generated rationales on REV (ours) ------ ###
echo "Experiment for model generated rationales on REV (ours)" >> log/rev.txt
for data_name in gpt-4_demo=2_raw=True gpt-3.5-turbo_demo=2_raw=True Llama-2-7b-hf_demo=2_raw=True flan-t5-large_demo=2_raw=True
do 
    if [ ! -e "data/processed_datasets/strategyqa_model_rationale/${data_name}/vocab_format=gs_ng=2_mf=1_mt=10000_r=1.pt" ]; then
        python scripts/generate_vocabs.py \
                --dataset-dir data/processed_datasets/strategyqa_model_rationale/${data_name} \
                --rationale-format gs \
                --rationale-only
    fi
    if [ ! -d "/scratch/ylu130/project/REV_reimpl/ckpt/fasttext-strategyqa_model_rationale_gs_1_${data_name}" ]; then
        python steps/train_rev_model.py \
                --task-name fasttext-strategyqa_model_rationale \
                --rationale-format gs \
                --data-name $data_name
    fi  

    if [ ! -d "/scratch/ylu130/project/REV_reimpl/ckpt/t5-strategyqa_model_rationale_gs_1_none_mask_${data_name}" ]; then
        python steps/train_rev_model.py \
                --task-name t5-strategyqa_model_rationale \
                --rationale-format gs \
                --data-name $data_name
    fi
    
    echo "Evaluating data: $data_name" >> log/rev.txt
    python steps/eval_rev_with_model.py \
            --dataset-dir data/processed_datasets/strategyqa_model_rationale/${data_name} \
            --model-dir /scratch/ylu130/project/REV_reimpl/ckpt/t5-strategyqa_model_rationale_gs_1_none_mask_${data_name} \
            --rationale-format gs >> log/rev.txt
done 

### ------ Experiments for model generated rationales on REV (original) ------ ###
echo "Experiments for model generated rationales on REV (original)" >> log/rev.txt
for data_name in gpt-4_demo=2_raw=True gpt-3.5-turbo_demo=2_raw=True Llama-2-7b-hf_demo=2_raw=True flan-t5-large_demo=2_raw=True
do 
    if [ ! -e "data/processed_datasets/strategyqa_model_rationale/${data_name}/vocab_format=gs_ng=2_mf=1_mt=10000_r=1.pt" ]; then
        python scripts/generate_vocabs.py \
                --dataset-dir data/processed_datasets/strategyqa_model_rationale/${data_name} \
                --rationale-format gs \
                --rationale-only
    fi
    if [ ! -d "/scratch/ylu130/project/REV_reimpl/ckpt/fasttext-strategyqa_model_rationale_gs_1_${data_name}" ]; then
        python steps/train_rev_model.py \
                --task-name fasttext-strategyqa_model_rationale \
                --rationale-format gs \
                --data-name $data_name
    fi  

    if [ ! -d "/scratch/ylu130/project/REV_reimpl/ckpt/t5-strategyqa_model_rationale_gs_1_none_mask_rationale_only_${data_name}" ]; then
        python steps/train_rev_model.py \
                --task-name t5-strategyqa_model_rationale \
                --rationale-format gs \
                --data-name $data_name \
                --rationale-only
    fi
    
    echo "Evaluating data: $data_name" >> log/rev.txt
    python steps/eval_rev_with_model.py \
            --dataset-dir data/processed_datasets/strategyqa_model_rationale/${data_name} \
            --model-dir /scratch/ylu130/project/REV_reimpl/ckpt/t5-strategyqa_model_rationale_gs_1_none_mask_rationale_only_${data_name} \
            --rationale-format gs \
            --rationale-only >> log/rev.txt
done 