### ------ Experiments for generating gpt4 and gpt3.5 rationales ------ ###
# echo "Experiments for generating gpt4 and gpt3.5 rationales"
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
#         --split test

# python scripts/generate_rationales.py \
#         --dataset-dir Zhengping/strategyqa_custom_split \
#         --model-name gpt-4 \
#         --demonstration-num 2 \
#         --output-dir Yining/generated_rationales/strategyqa \
#         --use-raw-model \
#         --split test

### ------ Experiments for generating llama and flan-t5 rationales ------ ###
# echo "Experiments for generating llama and flan-t5 rationales"
# for model in meta-llama/Llama-2-7b-hf google/flan-t5-large
# do 
#         for split in train validation
#         do
#             python scripts/generate_rationales.py \
#                     --dataset-dir Zhengping/strategyqa_custom_split \
#                     --model-name $model \
#                     --demonstration-num 2 \
#                     --output-dir Yining/generated_rationales/strategyqa \
#                     --use-raw-model \
#                     --split $split \
#                     --batch-size 4 
#         done
# done

# for model in  meta-llama/Llama-2-7b-hf google/flan-t5-large
# do 
#         python scripts/generate_rationales.py \
#                 --dataset-dir Zhengping/strategyqa_custom_split \
#                 --model-name $model \
#                 --demonstration-num 2 \
#                 --output-dir Yining/generated_rationales/strategyqa \
#                 --use-raw-model \
#                 --split test \
#                 --batch-size 4
# done