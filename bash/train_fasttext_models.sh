cd /home/ylu130/workspace/REV-reimpl

for rationale_format in 'g' 'l' 's' 'gls' 'gs' 'ls' 'gl' 'n'
do
    python steps/train_rev_model.py --task-name fasttext-strategyqa --rationale-format $rationale_format
done