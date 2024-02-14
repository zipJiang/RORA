import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# deberta, lambda = 10
delta = [0.1, 0.05, 0.01, 0.005, 0.001]
g = [0.041, -0.017, 0.039, 0.08, 0.079]
# g= [0.362, 0.381, 0.337, 0.404, 0.288]
gl = [0.027, 0.028, 0.038, 0.054, 0.090]
# gl= [0.179, 0.348, 0.3, 0.35, 0.385]
vacuous = [0, -0.013, -0.009, 0.014, 0.070]
# vacuous = [0.051, 0.026, 0.046, 0.048, 0.285]
l = [0.01, 0.02, 0.008, -0.001, -0.010]
# l = [0.013, 0.024, 0.059, 0.025, 0.063]

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
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

# ax1.plot(data['Leakage Detection Threshold'], data['gold'], marker='o', label='Gold', linewidth=2, markersize=8)
# ax1.plot(data['Leakage Detection Threshold'], data['gold + leaky'], marker='X', label='Gold + Leaky', linewidth=2, markersize=8)
# ax1.plot(data['Leakage Detection Threshold'], data['vacuous'], marker='s', label='Vacuous', linewidth=2, markersize=8)
# ax1.plot(data['Leakage Detection Threshold'], data['leaky'], marker='^', label='Leaky', linewidth=2, markersize=8)

# ax1.set_xlabel('Leakage Detection Threshold')
# ax1.set_ylabel('RORA')



plt.plot(data['Leakage Detection Threshold'], data['gold'], marker='o', label='Gold', linewidth=2, markersize=8)
plt.plot(data['Leakage Detection Threshold'], data['gold + leaky'], marker='X', label='Gold + Leaky', linewidth=2, markersize=8)
plt.plot(data['Leakage Detection Threshold'], data['vacuous'], marker='s', label='Vacuous', linewidth=2, markersize=8)
plt.plot(data['Leakage Detection Threshold'], data['leaky'], marker='^', label='Leaky', linewidth=2, markersize=8)

plt.xlabel('Leakage Detection Threshold')
plt.ylabel('RORA')
# reduce the number of ticks in y-axis
plt.yticks([-0.1, 0, 0.1])
# increase the size of the x-sticks and y-sticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# rename x-sticks
plt.xticks([0.1, 0.05, 0.01, 0.005, 0.001], ['0.001', '0.005', '', '', '0.1'])
# remove legend outside the plot
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
# increase the size of the x-sticks and y-sticks
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# rename x-sticks
plt.xticks([0.1, 0.05, 0.01, 0.005, 0.001], ['0.001', '0.005', '', '', '0.1'])
# reduce legend box size
plt.legend()
plt.savefig('threshold_model.pdf', format='pdf', bbox_inches='tight', dpi=800)
