import seaborn as sns
from scipy.stats import chi2_contingency
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# violinplot.pdf
df = pd.read_csv('data_prepare/output/main_result_0630.csv', index_col = 0)

sns.violinplot(x='label_2', y='dfr', data=df, inner=None)
sns.stripplot(x='label_2', y='dfr', data=df, jitter=True, color='black', alpha=1)

plt.title('Pirate Plot')
plt.xlabel('Label')
plt.ylabel('dfr')
plt.ylim(-0.2, 1.2)
fig = plt.gcf()
fig.set_size_inches(10, 10)

plt.savefig('figures/violinplot.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.show()

contingency_table = pd.crosstab(df['dfr'], df['label_2'])
chi2, p_value, dof, expected = chi2_contingency(contingency_table)
print('p_value:', p_value)

# compare.csv
df = pd.read_csv('figures/compare.csv', index_col=0)
columns = ['hl_loss', 'log_cosh_loss', 'mae_loss', 'mse_loss', 'huber_loss']

# mse
data_0 = df[(df['indicator'] == 'mse') & (df['dataset'] == 0)]
data_1 = df[(df['indicator'] == 'mse') & (df['dataset'] == 1)]
# hl value
# data_0 = df[(df['indicator'] == 'HL_value') & (df['dataset'] == 0)]
# data_1 = df[(df['indicator'] == 'HL_value') & (df['dataset'] == 1)]
# p value
# data_0 = df[(df['indicator'] == 'p_val') & (df['dataset'] == 0)]
# data_1 = df[(df['indicator'] == 'p_val') & (df['dataset'] == 1)]

values_0 = data_0[columns].values.flatten()
values_1 = data_1[columns].values.flatten()

bar_width = 0.35
index = np.arange(len(columns))

fig, ax = plt.subplots(figsize=(10, 6))

bar_0 = ax.barh(index, values_0, bar_width, label='Dataset 600')
bar_1 = ax.barh(index + bar_width, values_1, bar_width, label='Dataset 1000')

for rect in bar_0:
    width = rect.get_width()
    ax.text(width + 0.001, rect.get_y() + rect.get_height() / 2, f'{width:.5f}', ha='left', va='center')

for rect in bar_1:
    width = rect.get_width()
    ax.text(width + 0.001, rect.get_y() + rect.get_height() / 2, f'{width:.5f}', ha='left', va='center')

ax.set_xlabel('Values')
ax.set_ylabel('Metrics')
ax.set_title('Comparison of Values')
ax.set_yticks(index + bar_width / 2)
ax.set_yticklabels(columns)
ax.legend()

ax.set_xlim(0, 0.0175)
# ax.set_xlim(0, 3.8)
# ax.set_xlim(0.9, 1)

plt.tight_layout()

plt.savefig('figures/mse.pdf', format='pdf', dpi=300, bbox_inches='tight')
# plt.savefig('figures/hl_value.pdf', format='pdf', dpi=300, bbox_inches='tight')
# plt.savefig('figures/p_value.pdf', format='pdf', dpi=300, bbox_inches='tight')

plt.show()






