import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd
import numpy as np
# deberta, lambda = 10
delta = [0.1, 0.05, 0.01, 0.005, 0.001]
# g = [0.041, -0.017, 0.039, 0.08, 0.079]
g= [0.362, 0.381, 0.337, 0.404, 0.288]
# gl = [0.027, 0.028, 0.038, 0.054, 0.090]
gl= [0.179, 0.348, 0.3, 0.35, 0.385]
# vacuous = [0, -0.013, -0.009, 0.014, 0.070]
vacuous = [0.051, 0.026, 0.046, 0.048, 0.285]
# l = [0.01, 0.02, 0.008, -0.001, -0.010]
l = [0.013, 0.024, 0.059, 0.025, 0.063]

# gpt4 = [0.595, 0.56, 0.587, 0.474, 0.442]
gpt4 = [0.442, 0.474, 0.587, 0.56, 0.595]
# gpt3 = [0.484, 0.465, 0.425, 0.459, 0.434]
gpt3 = [0.434, 0.459, 0.425, 0.465, 0.484]
# llama2 = [0.316, 0.381, 0.3, 0.232, 0.235]
llama2 = [0.235, 0.232, 0.3, 0.381, 0.316]
# flant5 = [0.137, 0.096, 0.157, 0.132, 0.123]
flant5 = [0.123, 0.132, 0.157, 0.096, 0.137]


data = pd.DataFrame({'Leakage Detection Threshold': delta, 'gold': g, 'gold + leaky': gl, 'vacuous': vacuous, 'leaky': l, 'gpt-4': gpt4, 'gpt-3': gpt3, 'llama-2': llama2, 'flant5': flant5})

sns.set_theme(style="white")
plt.figure(figsize=(6, 3))
plt.plot(data['Leakage Detection Threshold'], data['gold'], marker='o', label='Gold', linewidth=2, markersize=8)
plt.plot(data['Leakage Detection Threshold'], data['gold + leaky'], marker='X', label='Gold + Leaky', linewidth=2, markersize=8)
plt.plot(data['Leakage Detection Threshold'], data['vacuous'], marker='s', label='Vacuous', linewidth=2, markersize=8)
plt.plot(data['Leakage Detection Threshold'], data['leaky'], marker='^', label='Leaky', linewidth=2, markersize=8)

plt.xlabel('Leakage Detection Threshold')
plt.ylabel('RORA')
plt.yticks([-0.5, 0, 0.5])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xticks([0.1, 0.05, 0.01, 0.005, 0.001], ['0.001', '0.005', '', '', '0.1'])
plt.legend()
plt.savefig('threshold_synthetic.pdf', format='pdf', bbox_inches='tight', dpi=800)


sns.set_theme(style="white")
plt.figure(figsize=(6, 3))
plt.plot(data['Leakage Detection Threshold'], data['gpt-4'], label='GPT-4', linewidth=3, linestyle='-')
plt.plot(data['Leakage Detection Threshold'], data['gpt-3'], label='GPT-3.5', linewidth=3, linestyle='--')
plt.plot(data['Leakage Detection Threshold'], data['llama-2'], label='Llama2-7B', linewidth=3, linestyle='-.')
plt.plot(data['Leakage Detection Threshold'], data['flant5'], label='Flan-T5 Large', linewidth=3, linestyle=':')

plt.xlabel('Leakage Detection Threshold')
plt.ylabel('RORA')
plt.yticks([-0.5, 0, 0.5])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xticks([0.1, 0.05, 0.01, 0.005, 0.001], ['0.001', '0.005', '', '', '0.1'])
plt.legend()
plt.savefig('threshold_model.pdf', format='pdf', bbox_inches='tight', dpi=800)

##############################################
# lambda, delta = 0.005

lambd = [0, 1, 5, 10, 50, 100, 500, 1000]
lambd = np.log10(lambd)
g = [0.406, 0.334, 0.389, 0.381, 0.032, 0.012, 0.003, 0.002]
gl = [0.372, 0.296, 0.338, 0.348, 0.049, 0.017, 0.002, 0.004]
s = [-0.023, 0.055, 0.043, 0.026, 0.012, 0.006, 0.002, -0.001]
l = [0.036, 0.06, 0.012, 0.024, 0.012, 0.004, 0.0, 0.002]

data = pd.DataFrame({'IRM Coefficient': lambd, 'gold': g, 'gold + leaky': gl, 'vacuous': s, 'leaky': l})
sns.set_theme(style="white")
plt.figure(figsize=(6, 3))
plt.plot(data['IRM Coefficient'], data['gold'], marker='o', label='Gold', alpha=0.7)
plt.plot(data['IRM Coefficient'], data['gold + leaky'], marker='X', label='Gold + Leaky',  alpha=0.7)
plt.plot(data['IRM Coefficient'], data['vacuous'], marker='s', label='Vacuous', alpha=0.7)
plt.plot(data['IRM Coefficient'], data['leaky'], marker='^', label='Leaky',  alpha=0.7)
plt.xlabel('IRM Coefficient')
plt.ylabel('RORA')
plt.yticks([0, 0, 0.5])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()

gpt4 = [0.568, 0.515, 0.517, 0.474, 0.027, 0.017, 0.001, 0.003]
gpt3 = [0.5, 0.478, 0.441, 0.459, 0.022, 0.019, 0.0, 0.0]
llama2 = [0.222, 0.301, 0.263, 0.232, 0.019, 0.011, 0.001, -0.001]
flant5 = [0.052, 0.102, 0.101, 0.132, 0.011, 0.007, -0.001, 0.0]
data = pd.DataFrame({'IRM Coefficient': lambd, 'gpt-4': gpt4, 'gpt-3': gpt3, 'llama-2': llama2, 'flant5': flant5})
sns.set_theme(style="white")
plt.figure(figsize=(6, 3))
plt.plot(data['IRM Coefficient'], data['gpt-4'], label='GPT-4', linestyle='-')
plt.plot(data['IRM Coefficient'], data['gpt-3'], label='GPT-3.5', linestyle='--')
plt.plot(data['IRM Coefficient'], data['llama-2'], label='Llama2-7B', linestyle='-.')
plt.plot(data['IRM Coefficient'], data['flant5'], label='Flan-T5 Large',  linestyle=':')
plt.xlabel('IRM Coefficient')
plt.ylabel('RORA')
plt.yticks([0, 0, 0.5])
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend()


##############################################

# delta = [0.1, 0.05, 0.01, 0.005, 0.001]
delta = [0.001, 0.005, 0.01, 0.05, 0.1]
# g = [0.041, -0.017, 0.039, 0.08, 0.079]
g= [0.362, 0.381, 0.337, 0.404, 0.288]
# gl = [0.027, 0.028, 0.038, 0.054, 0.090]
gl= [0.179, 0.348, 0.3, 0.35, 0.385]
# vacuous = [0, -0.013, -0.009, 0.014, 0.070]
vacuous = [0.051, 0.026, 0.046, 0.048, 0.285]
# l = [0.01, 0.02, 0.008, -0.001, -0.010]
l = [0.013, 0.024, 0.059, 0.025, 0.063]

# gpt4 = [0.595, 0.56, 0.587, 0.474, 0.442]
gpt4 = [0.442, 0.474, 0.587, 0.56, 0.595]
# gpt3 = [0.484, 0.465, 0.425, 0.459, 0.434]
gpt3 = [0.434, 0.459, 0.425, 0.465, 0.484]
# llama2 = [0.316, 0.381, 0.3, 0.232, 0.235]
llama2 = [0.235, 0.232, 0.3, 0.381, 0.316]
# flant5 = [0.137, 0.096, 0.157, 0.132, 0.123]
flant5 = [0.123, 0.132, 0.157, 0.096, 0.137]


data1 = pd.DataFrame({'Leakage Detection Threshold': delta, 'gold': g, 'gold + leaky': gl, 'vacuous': vacuous, 'leaky': l, 'gpt-4': gpt4, 'gpt-3': gpt3, 'llama-2': llama2, 'flant5': flant5})

lambd = [0, 1, 5, 10, 50, 100, 500, 1000]
lambd = np.log10(lambd)
g = [0.406, 0.334, 0.389, 0.381, 0.032, 0.012, 0.003, 0.002]
gl = [0.372, 0.296, 0.338, 0.348, 0.049, 0.017, 0.002, 0.004]
s = [-0.023, 0.055, 0.043, 0.026, 0.012, 0.006, 0.002, -0.001]
l = [0.036, 0.06, 0.012, 0.024, 0.012, 0.004, 0.0, 0.002]
gpt4 = [0.568, 0.515, 0.517, 0.474, 0.027, 0.017, 0.001, 0.003]
gpt3 = [0.5, 0.478, 0.441, 0.459, 0.022, 0.019, 0.0, 0.0]
llama2 = [0.222, 0.301, 0.263, 0.232, 0.019, 0.011, 0.001, -0.001]
flant5 = [0.052, 0.102, 0.101, 0.132, 0.011, 0.007, -0.001, 0.0]

data2 = pd.DataFrame({'IRM Coefficient': lambd, 'gold': g, 'gold + leaky': gl, 'vacuous': s, 'leaky': l, 'gpt-4': gpt4, 'gpt-3': gpt3, 'llama-2': llama2, 'flant5': flant5})

sns.set_theme(style="darkgrid")
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey='row')
fig.set_size_inches(10, 5)
fig.subplots_adjust(wspace=0.3, hspace=0.3)

ax1.plot(data1['Leakage Detection Threshold'], data1['gold'], marker='o', label='Gold', linewidth=2, markersize=8)
ax1.plot(data1['Leakage Detection Threshold'], data1['gold + leaky'], marker='X', label='Gold + Leaky', linewidth=2, markersize=8)
ax1.plot(data1['Leakage Detection Threshold'], data1['vacuous'], marker='s', label='Vacuous', linewidth=2, markersize=8)
ax1.plot(data1['Leakage Detection Threshold'], data1['leaky'], marker='^', label='Leaky', linewidth=2, markersize=8)
ax1.set_xlabel('Leakage Detection Threshold')
ax1.set_ylabel('RORA')
ax1.set_yticks([0, 0, 0.5])
ax1.set_xticks([0.001, 0.005, 0.01, 0.05, 0.1], ['0.001', '', '', '0.05', '0.1'])
# increase the font size of the x and y ticks
ax1.tick_params(axis='both', which='major', labelsize=14)


ax2.plot(data1['Leakage Detection Threshold'], data1['gpt-4'], label='GPT-4', linestyle='-', linewidth=3)
ax2.plot(data1['Leakage Detection Threshold'], data1['gpt-3'], label='GPT-3.5', linestyle='--', linewidth=3)
ax2.plot(data1['Leakage Detection Threshold'], data1['llama-2'], label='Llama2-7B', linestyle='-.', linewidth=3)
ax2.plot(data1['Leakage Detection Threshold'], data1['flant5'], label='Flan-T5 Large', linestyle=':', linewidth=3)
ax2.set_xlabel('Leakage Detection Threshold')
ax2.set_xticks([0.001, 0.005, 0.01, 0.05, 0.1], ['0.001', '', '', '0.05', '0.1'])
ax2.tick_params(axis='both', which='major', labelsize=14)

ax3.plot(data2['IRM Coefficient'], data2['gold'], marker='o', label='Gold', linewidth=2, markersize=8)
ax3.plot(data2['IRM Coefficient'], data2['gold + leaky'], marker='X', label='Gold + Leaky', linewidth=2, markersize=8)
ax3.plot(data2['IRM Coefficient'], data2['vacuous'], marker='s', label='Vacuous', linewidth=2, markersize=8)
ax3.plot(data2['IRM Coefficient'], data2['leaky'], marker='^', label='Leaky', linewidth=2, markersize=8)
ax3.set_xlabel('IRM Regularization Parameter (log10)')
ax3.set_ylabel('RORA')
ax3.set_yticks([0, 0, 0.5])
ax3.tick_params(axis='both', which='major', labelsize=14)

ax4.plot(data2['IRM Coefficient'], data2['gpt-4'], label='GPT-4', linestyle='-', linewidth=3)
ax4.plot(data2['IRM Coefficient'], data2['gpt-3'], label='GPT-3.5', linestyle='--', linewidth=3)
ax4.plot(data2['IRM Coefficient'], data2['llama-2'], label='Llama2-7B', linestyle='-.', linewidth=3)
ax4.plot(data2['IRM Coefficient'], data2['flant5'], label='Flan-T5 Large',  linestyle=':', linewidth=3)
ax4.set_xlabel('IRM Regularization Parameter (log10)')
ax4.tick_params(axis='both', which='major', labelsize=14)

# make ax1 and ax3 share the same legend, and ax2 and ax4 share the same legend. Put the legend above the subplots
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=2)
ax2.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), ncol=2)

# tight_layout() adjusts the subplots to fit into the figure area.
plt.tight_layout()

plt.savefig('sensitivity_analysis.pdf', format='pdf', bbox_inches='tight', dpi=1500)


