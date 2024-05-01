import matplotlib.pyplot as plt

# # Data
# items = ['Logistic Regression (Original)', 'Logistic Regression (Embeddings)', 'Booster on Embeddings']
# fairness_values = [0.996, 0.9848, 0.9842]

# # Create bar plot
# plt.figure(figsize=(8, 6))
# plt.bar(items, fairness_values, color='skyblue')
# plt.xlabel('Model')
# plt.ylabel('Individual Fairness Score')
# plt.title('Individual Fairness of 3 Models for ADULT dataset')
# plt.ylim(0, 1)  # Set y-axis limit from 0 to 1
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()

# # Data
# items = ['Logistic Regression (Original)', 'Logistic Regression (Embeddings)', 'Booster on Embeddings']
# fairness_values = [0.9047, 0.9937, 0.9842]

# # Create bar plot
# plt.figure(figsize=(8, 6))
# plt.bar(items, fairness_values, color='skyblue')
# plt.xlabel('Model')
# plt.ylabel('Individual Fairness Score')
# plt.title('Individual Fairness of 3 Models for COMPAS dataset')
# plt.ylim(0, 1)  # Set y-axis limit from 0 to 1
# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.show()

import numpy as np

# Data
models = ['Logistic Regression (Original)', 'Logistic Regression (Embeddings)', 'Booster on Embeddings']
accuracy_values = [0.84, 0.79, 0.79]
weighted_f1_values = [0.83, 0.74, 0.75]

x = np.arange(len(models))
bar_width = 0.35

# Create bar plot
fig, ax = plt.subplots(figsize=(10, 6))
accuracy_bars = ax.bar(x - bar_width/2, accuracy_values, bar_width, label='Accuracy', color='skyblue')
f1_bars = ax.bar(x + bar_width/2, weighted_f1_values, bar_width, label='Weighted F1', color='lightgreen')

# Add labels, title, and legend
ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Accuracy and Weighted F1 Score Comparison on Adult')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Show values on top of bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(accuracy_bars)
autolabel(f1_bars)

plt.tight_layout()
plt.show()


# Data
models = ['Logistic Regression (Original)', 'Logistic Regression (Embeddings)', 'Booster on Embeddings']
accuracy_values = [0.59, 0.56, 0.60]
weighted_f1_values = [0.58, 0.54, 0.59]

x = np.arange(len(models))
bar_width = 0.35

# Create bar plot
fig, ax = plt.subplots(figsize=(10, 6))
accuracy_bars = ax.bar(x - bar_width/2, accuracy_values, bar_width, label='Accuracy', color='skyblue')
f1_bars = ax.bar(x + bar_width/2, weighted_f1_values, bar_width, label='Weighted F1', color='lightgreen')

# Add labels, title, and legend
ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Accuracy and Weighted F1 Score Comparison on COMPAS')
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend()

# Show values on top of bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate('{}'.format(height),
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(accuracy_bars)
autolabel(f1_bars)

plt.tight_layout()
plt.show()