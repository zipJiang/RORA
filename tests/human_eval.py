import pandas as pd
import pickle
import json
import numpy as np
import transformers

with open('/home/ylu130/workspace/REV-reimpl/tests/results/baseline_to_human_map.pkl', 'rb') as f:
    human_eval_map = pickle.load(f)

with open('/home/ylu130/workspace/REV-reimpl/tests/results/baseline_to_irm_map.pkl', 'rb') as f:
    irm_map = pickle.load(f)

heval_df = pd.read_csv('/home/ylu130/workspace/REV-reimpl/tests/results/rationale-evaluation/heval.csv')
pilot_heval_df = pd.read_csv('/home/ylu130/workspace/REV-reimpl/tests/results/rationale-evaluation/pilot.heval.csv')

human_eval = pd.concat([pilot_heval_df, heval_df])

# replace all "{}" in Answer.levelOfSupport column with -1
human_eval['Answer.levelOfSupport'] = human_eval['Answer.levelOfSupport'].replace('{}', -1)

# group by HITId to find the average Answer.levelOfSupport
human_eval['Answer.levelOfSupport'] = human_eval['Answer.levelOfSupport'].astype(int)
human_eval = human_eval[['HITId', 'Answer.levelOfSupport', 'Answer.explanation', 'Answer.question', 'Answer.modelType']]
human_eval = human_eval.groupby('HITId').agg({'Answer.levelOfSupport': 'mean', 'Answer.explanation': 'first', 'Answer.question': 'first', 'Answer.modelType': 'first'}).reset_index()

# merge with human_eval_map to get the baseline and irm
human_eval_map = pd.DataFrame(human_eval_map.items(), columns=['question', 'baseline'])
# remove the "question: " prefix in both columns
human_eval_map['question'] = human_eval_map['question'].str.replace('question: ', '')
human_eval_map['baseline'] = human_eval_map['baseline'].str.replace('question: ', '')

# replace "&amp;" with "&" in Answer.question column
human_eval['Answer.question'] = human_eval['Answer.question'].str.replace('&amp;', '&')
human_eval = human_eval.merge(human_eval_map, left_on='Answer.question', right_on='question', how='left')


def get_rora(path, model_name):
    with open(path, 'r') as f:
        rora = json.load(f)

    baselilne_loss = rora['baseline']['elementwise_loss']

    rora_loss = rora['rev']['elementwise_loss']

    rora_list = [baseline - rev for baseline, rev in zip(baselilne_loss, rora_loss)]
    # load deberta_large tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')
    questions = tokenizer.batch_decode(rora['baseline']['input_ids'], skip_special_tokens=True)

    assert np.mean(rora_list) - rora['overall_result'] < 1e-2

    return pd.DataFrame({'question': questions, 'rora': rora_list, 'model': model_name})

gpt4 = get_rora('/home/ylu130/workspace/REV-reimpl/tests/results/strategyqa-elementwise-eval/rev=deberta_lr=0.00001_rm=fasttext_format=gpt4_ng=2_mf=1_mt=10000_th=0.001_irm=10.0.json', 'gpt4')
gpt3 = get_rora('/home/ylu130/workspace/REV-reimpl/tests/results/strategyqa-elementwise-eval/rev=deberta_lr=0.00001_rm=fasttext_format=gpt3_ng=2_mf=1_mt=10000_th=0.001_irm=10.0.json', 'gpt3')
llama2 = get_rora('/home/ylu130/workspace/REV-reimpl/tests/results/strategyqa-elementwise-eval/rev=deberta_lr=0.00001_rm=fasttext_format=llama2_ng=2_mf=1_mt=10000_th=0.001_irm=10.0.json', 'llama2')
flan = get_rora('/home/ylu130/workspace/REV-reimpl/tests/results/strategyqa-elementwise-eval/rev=deberta_lr=0.00001_rm=fasttext_format=flan_ng=2_mf=1_mt=10000_th=0.001_irm=10.0.json', 'flan')

# concat all rora dfs
rora = pd.concat([gpt4, gpt3, llama2, flan]).reset_index(drop=True)
rora['question'] = rora['question'].str.replace('question: ', '')
rora['question'] = rora['question'].str.replace(' rationale:', '')

# merge with human_eval on question and model type
human_eval = human_eval.merge(rora, left_on=['question', 'Answer.modelType'], right_on=['question', 'model'], how='left')

# check how many NaNs in rora column
print(human_eval['rora'].isna().sum())

import matplotlib.pyplot as plt
import seaborn as sns

# draw four subplots for each model within one row
sns.set_theme(style="white")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
# in each subplot, plot the distribution of support vs. rora and fit a linear line
# different color for different model
for i, model in enumerate(human_eval['Answer.modelType'].unique()):
    if model == 'gpt4':
        color = 'royalblue'
        title = 'GPT-4'
    elif model == 'gpt3':
        color = 'darkorange'
        title = 'GPT-3.5'
    elif model == 'llama2':
        color = 'seagreen'
        title = 'Llama2-7B'
    else:
        color = 'tomato'
        title = 'Flan-T5 Large'
    sns.regplot(x='Answer.levelOfSupport', y='rora', data=human_eval[human_eval['Answer.modelType'] == model], ax=axes[i], color=color, scatter_kws={'alpha':0.8}, line_kws={'linewidth': 4})
    axes[i].set_title(title, fontsize=16)
    axes[i].set_xlabel('Human Evaluation', fontsize=16)
    axes[i].set_ylabel('RORA', fontsize=16)
    axes[i].set_xlim(-1.1, 3.1)
    axes[i].set_ylim(-2, 2)
    # set ticks for x and y axis
    axes[i].set_xticks(np.arange(-1, 4, 1))
    axes[i].set_yticks(np.arange(-2, 3, 1))
    # increase the font size of x and y ticks
    axes[i].tick_params(axis='x', labelsize=16)
    axes[i].tick_params(axis='y', labelsize=16)

# only show the y label in the first subplot
axes[1].set_ylabel('')
axes[2].set_ylabel('')
axes[3].set_ylabel('')

plt.savefig('dist.pdf', dpi=2500, bbox_inches='tight')