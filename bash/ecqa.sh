# for split in train validation test
# do 
#     python steps/rationale_preprocessing.py \
#             --data-handle yangdong/ecqa \
#             --split $split \
#             --write-to data/processed_datasets/ecqa
# done

for model in gpt2 t5-large
do 
    python steps/train_rationale_generator.py --task-name ecqa --model-name ${model}
done