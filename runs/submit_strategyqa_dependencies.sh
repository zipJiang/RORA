#!/bin/bash

set -e
set -i
set -x


# We submit the job satisfying dependency in the following way:

# first, we submit the job to calculate all previous dependencies
# for a given target, and then we submit the job to calculate the
# target itself.

# first submit job to create baseline_model

jid_raw=$(sbatch runs/run_make_strategyqa.sh raw_dataset g 0 | awk '{print $4}')
jid_baseline=$(sbatch --dependency=afterok:$jid_raw runs/run_make_strategyqa.sh baseline_model g 0 | awk '{print $4}')

# for rationale_format in g gl l s ; do
for rationale_format in llama2 flan s l g gl gpt4 gpt3; do
    jid_rev_dataset=$(sbatch --dependency=afterok:$jid_raw runs/run_make_strategyqa.sh rev_dataset $rationale_format 0 | awk '{print $4}')
    for irm_coefficient in 0 1 10 100 1000 10000 ; do
        sbatch --dependency=afterok:$jid_baseline:$jid_rev_dataset runs/run_make_strategyqa.sh report_file $rationale_format $irm_coefficient
    done
done
