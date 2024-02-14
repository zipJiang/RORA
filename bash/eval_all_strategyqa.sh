## ----- train evaluation model ----- ##
for rationale_format in gsl gls
do
    for irm in 10.0 100.0
    do
        for remove_threshold in 0.005 0.01
        do
            if [ ! -e "data/processed_datasets/strategyqa/vocab_format=${rationale_format}_ng=2_mf=1_mt=10000_r=1.pt" ]; then
            python scripts/generate_vocabs.py \
                --dataset-dir data/processed_datasets/strategyqa \
                --rationale-format $rationale_format \
                --rationale-only
            fi

            if [ ! -d "/scratch/ylu130/project/REV_reimpl/ckpt/fasttext-strategyqa_${rationale_format}_1" ]; then
            echo "Training FastText model with rationale format: ${rationale_format}"
            python steps/train_rev_model.py \
                --task-name fasttext-strategyqa \
                --rationale-format ${rationale_format}
            fi
            
            # train generator
            if [ ! -d "/scratch/ylu130/project/REV_reimpl/ckpt/generation/strategyqa_t5-base_${rationale_format}_${remove_threshold}" ]; then
                echo "Training generator with rationale format: ${rationale_format}"
                python steps/train_generator.py \
                        --task-name strategyqa \
                        --rationale-format ${rationale_format} \
                        --removal-threshold ${remove_threshold}
            fi
            # train irm model  
            if [ ! -d "/scratch/ylu130/project/REV_reimpl/ckpt/irm/strategyqa_microsoft::deberta-v3-large_${rationale_format}_${remove_threshold}_${irm}" ]; then
                echo "Training IRM evaluation model with rationale format ${rationale_format} and IRM coefficient ${irm}"
                python steps/train_irm_model.py \
                        --rationale-format ${rationale_format} \
                        --removal-threshold ${remove_threshold} \
                        --irm-coefficient ${irm} \
                        --model-name microsoft/deberta-v3-large \
                        --batch-size 8
            fi
        done
    done
done

## ------ Evaluation experiments for model-generated rationales on REV2 (deberta-v3-large) ------ ###
echo "Evaluation experiments for model-generated rationales on REV2 (deberta-v3-large)" >> log/rev2_eval.txt
for data_name in gpt-4_demo=2_raw=True gpt-3.5-turbo_demo=2_raw=True Llama-2-7b-hf_demo=2_raw=True flan-t5-large_demo=2_raw=True t5-large_demo=0_raw=False gpt2_demo=0_raw=False
do 
    for rationale_format in gsl gls
    do
        for irm in 10.0 100.0
        do 
            for remove_threshold in 0.005 0.01
            do
                echo "Evaluating rationale format: ${data_name} with remove threshold: ${remove_threshold} and IRM coefficient: ${irm} using ratioanle format: ${rationale_format}" >> log/rev2_eval.txt
                python steps/eval_rev_with_model.py \
                        --dataset-dir data/processed_datasets/strategyqa_model_rationale/${data_name} \
                        --model-dir /scratch/ylu130/project/REV_reimpl/ckpt/irm/strategyqa_microsoft::deberta-v3-large_${rationale_format}_${remove_threshold}_${irm} \
                        --rationale-format g >> log/rev2_eval.txt
            done
        done
    done
done

## ------ Evaluation experiments for synthetic rationales on REV2 (deberta-v3-large) ------ ###
echo "Evaluation experiments for synthetic rationales on REV2 (deberta-v3-large)" >> log/rev2_eval.txt
for data_name in g gl s l
do 
    for rationale_format in gsl gls
    do
        for irm in 10.0 100.0
        do 
            for remove_threshold in 0.005 0.01
            do
                echo "Evaluating rationale format: ${data_name} with remove threshold: ${remove_threshold} and IRM coefficient: ${irm} using ratioanle format: ${rationale_format}" >> log/rev2_eval.txt
                python steps/eval_rev_with_model.py \
                        --dataset-dir data/processed_datasets/strategyqa \
                        --model-dir /scratch/ylu130/project/REV_reimpl/ckpt/irm/strategyqa_microsoft::deberta-v3-large_${rationale_format}_${remove_threshold}_${irm} \
                        --rationale-format ${data_name} >> log/rev2_eval.txt
            done
        done
    done
done
