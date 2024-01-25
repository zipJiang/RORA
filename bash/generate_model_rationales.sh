# for split in train validation
# do
#     python scripts/generate_rationales.py \
#             --dataset-dir Zhengping/strategyqa_custom_split \
#             --model-name gpt-4 \
#             --demonstration-num 2 \
#             --output-dir Yining/generated_rationales/strategyqa \
#             --use-raw-model \
#             --split $split

#     python scripts/generate_rationales.py \
#             --dataset-dir Zhengping/strategyqa_custom_split \
#             --model-name gpt-3.5-turbo \
#             --demonstration-num 2 \
#             --output-dir Yining/generated_rationales/strategyqa \
#             --use-raw-model \
#             --split $split
# done

# python scripts/generate_rationales.py \
#         --dataset-dir Zhengping/strategyqa_custom_split \
#         --model-name gpt-3.5-turbo \
#         --demonstration-num 2 \
#         --output-dir Yining/generated_rationales/strategyqa \
#         --use-raw-model \
#         --split test \
#         --num-sample 200

for model in google/flan-t5-large meta-llama/Llama-2-7b-hf
do 
        for split in train validation
        do
            python scripts/generate_rationales.py \
                    --dataset-dir Zhengping/strategyqa_custom_split \
                    --model-name $model \
                    --demonstration-num 2 \
                    --output-dir Yining/generated_rationales/strategyqa \
                    --use-raw-model \
                    --split $split \
                    --batch-size 4 
        done
done

for model in google/flan-t5-large meta-llama/Llama-2-7b-hf
do 
        python scripts/generate_rationales.py \
                --dataset-dir Zhengping/strategyqa_custom_split \
                --model-name $model \
                --demonstration-num 2 \
                --output-dir Yining/generated_rationales/strategyqa \
                --use-raw-model \
                --split test \
                --batch-size 4 \
                --num-sample 200
done