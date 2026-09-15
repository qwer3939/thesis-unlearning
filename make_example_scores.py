import pandas as pd
import numpy as np

np.random.seed(1024)
n_users = 100
n_items = 50
rows = []
for u in range(n_users):
    for i in range(n_items):
        rows.append([u, i, round(np.random.random(), 6)])

df = pd.DataFrame(rows, columns=['user_id', 'item_id', 'pred_score'])
df.to_csv('example_scores.csv', index=False)
print('example_scores.csv saved:', len(df), 'rows')