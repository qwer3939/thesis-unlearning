import sys
import numpy as np
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else 'Data/Process/MovieLens/0.03'
THRESHOLDS = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90]

cols = ['user', 'item', 'label']
train_normal = pd.read_csv(path + '/train_normal.csv')[cols]
train_random = pd.read_csv(path + '/train_random.csv')[cols]
valid = pd.read_csv(path + '/valid.csv')[cols]
test = pd.read_csv(path + '/test.csv')[cols]

n_users = 1 + max(train_normal['user'].max(), train_random['user'].max(),
                  valid['user'].max(), test['user'].max())
n_items = 1 + max(train_normal['item'].max(), train_random['item'].max(),
                  valid['item'].max(), test['item'].max())

normal_pos = train_normal[train_normal['label'] > 0][cols].values
unlearn_pos = train_random[train_random['label'] > 0][cols].values
all_pos = np.concatenate([normal_pos, unlearn_pos], axis=0)

deg_all = np.concatenate([
    np.bincount(all_pos[:, 0].astype(int), minlength=n_users),
    np.bincount(all_pos[:, 1].astype(int), minlength=n_items)]).astype(float)
deg_clean = np.concatenate([
    np.bincount(normal_pos[:, 0].astype(int), minlength=n_users),
    np.bincount(normal_pos[:, 1].astype(int), minlength=n_items)]).astype(float)

changed_ratio = 1 - deg_clean / (deg_all + 1e-9)
changed_ratio[deg_all == 0] = 0.0

unlearned_idx = np.concatenate([
    np.unique(unlearn_pos[:, 0].astype(int)),
    np.unique(unlearn_pos[:, 1].astype(int)) + n_users])

n_normal = len(train_normal)
print('user %d / item %d / normal %d / fake %d' % (n_users, n_items, n_normal, len(train_random)))
print('touched nodes %d (user %d, item %d)' % (
    len(unlearned_idx), (unlearned_idx < n_users).sum(), (unlearned_idx >= n_users).sum()))
r = changed_ratio[unlearned_idx]
print('fake-share  mean %.3f  median %.3f  min %.3f  max %.3f' % (
    r.mean(), np.median(r), r.min(), r.max()))
print()
print(' thr  |  items | changed_data |  vs normal')
print('------+--------+--------------+-----------')
for t in THRESHOLDS:
    ui = unlearned_idx[changed_ratio[unlearned_idx] > t]
    cu = ui[ui < n_users]
    ci = ui[ui >= n_users] - n_users
    ch = pd.concat([train_normal[train_normal['user'].isin(cu)],
                    train_normal[train_normal['item'].isin(ci)]],
                   axis=0, ignore_index=True).drop_duplicates(keep='last', ignore_index=True)
    print(' %4.2f | %6d | %12d | %8.1f%%' % (t, len(ci), len(ch), 100.0 * len(ch) / n_normal))
