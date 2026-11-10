import os
import pandas as pd

CH = ['0.01', '0.10', '0.20', '0.30', '0.50', '0.70']
NCH = {'0.01': 476191, '0.10': 208812, '0.20': 109627,
       '0.30': 54841, '0.50': 27346, '0.70': 10511}
T = 'ifru_trace_filler150_attack0.03_lr10000.csv'
ROOT = 'results_ch'

print('ratio changed_rows  setup   loop | HR0_ep  AUC@HR0 | HR_fin  AUC_fin | resid_0   resid_end | p_end')
print('-' * 108)
for c in CH:
    f = os.path.join(ROOT, c, T)
    if not os.path.exists(f):
        print('%5s  (no file: %s)' % (c, f))
        continue
    d = pd.read_csv(f)
    setup = d['cumulative_time'].iloc[0] - d['step_time'].iloc[0]
    loop = d['cumulative_time'].iloc[-1] - d['cumulative_time'].iloc[0]
    zero = d.index[d['hr10_clean'] <= 1e-9]
    if len(zero):
        i0 = zero[0]
        ep0 = '%6d' % int(d.loc[i0, 'epoch'])
        auc0 = '%8.4f' % d.loc[i0, 'test_auc_clean']
    else:
        imin = d['hr10_clean'].idxmin()
        ep0 = '  none'
        auc0 = '%8.4f' % d.loc[imin, 'test_auc_clean']
    print('%5s %12d %6.0f %6.1f | %s %s | %6.3f%% %8.4f | %9.1f %10.1f | %.3g' % (
        c, NCH[c], setup, loop, ep0, auc0,
        100 * d['hr10_clean'].iloc[-1], d['test_auc_clean'].iloc[-1],
        d['residual'].iloc[0], d['residual'].iloc[-1], d['p_norm'].iloc[-1]))

print()
print('residual trajectory (epoch 0 / 50 / 100 / 150 / 199)')
print('ratio       ep0       ep50      ep100      ep150      ep199   trend')
print('-' * 72)
for c in CH:
    f = os.path.join(ROOT, c, T)
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f)
    pts = [d['residual'].iloc[min(e, len(d) - 1)] for e in (0, 50, 100, 150, 199)]
    trend = 'DOWN' if pts[-1] < pts[0] else 'UP'
    print('%5s %10.1f %10.1f %10.1f %10.1f %10.1f   %s (x%.2f)' % (
        c, pts[0], pts[1], pts[2], pts[3], pts[4], trend, pts[-1] / max(pts[0], 1e-9)))

print()
print('AUC / HR trajectory at CH=0.30 (every 20 epochs)')
d = pd.read_csv(os.path.join(ROOT, '0.30', T))
print(d.iloc[::20][['epoch', 'hr10_clean', 'test_auc_clean', 'residual', 'p_norm']].to_string(index=False))
