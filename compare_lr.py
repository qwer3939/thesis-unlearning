import glob
import os
import pandas as pd

files = sorted(glob.glob('results/*_lam*.csv'))
if not files:
    raise SystemExit('!! results/ 에 *_lam*.csv 가 없습니다.')

rows = []
for f in files:
    d = pd.read_csv(f)
    lr = d['if_lr'].iloc[0]
    lam = d['lam'].iloc[0]
    r0, r1 = d['residual'].iloc[0], d['residual'].iloc[-1]
    rmin = d['residual'].min()
    zero = d.index[d['hr10_clean'] <= 1e-9]
    if len(zero):
        hr0_ep = int(d.loc[zero[0], 'epoch'])
        auc_at = d.loc[zero[0], 'test_auc_clean']
    else:
        hr0_ep = -1
        auc_at = d.loc[d['hr10_clean'].idxmin(), 'test_auc_clean']
    rows.append({
        'if_lr': lr, 'lam': lam,
        'resid_0': r0, 'resid_min': rmin, 'resid_end': r1,
        'ratio': r1 / max(r0, 1e-12),
        'hr0_ep': hr0_ep, 'auc_at_hr0': auc_at,
        'hr_end': 100 * d['hr10_clean'].iloc[-1],
        'auc_end': d['test_auc_clean'].iloc[-1],
        'p_end': d['p_norm'].iloc[-1],
        'loop_s': d['loop_time'].iloc[-1],
        'setup_s': d['setup_time'].iloc[0],
    })

t = pd.DataFrame(rows).sort_values('if_lr', ascending=False)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda v: '%.4g' % v)
print(t.to_string(index=False))
print()
print('resid ratio < 1 이면 잔차 감소 = 수렴 방향')
print('hr0_ep = -1 이면 타깃이 끝까지 0 미도달')
print()

best = t[t['ratio'] < 1]
if len(best):
    print('>> 잔차가 감소한 조합:')
    print(best[['if_lr', 'lam', 'resid_0', 'resid_end', 'ratio', 'hr_end', 'auc_end']].to_string(index=False))
else:
    print('>> 잔차가 감소한 조합 없음. 전부 발산.')

print()
print('=== 조합별 궤적 ===')
for f in files:
    d = pd.read_csv(f)
    print('--- if_lr=%g lam=%g ---' % (d['if_lr'].iloc[0], d['lam'].iloc[0]))
    print(d.iloc[::max(1, len(d) // 6)][['epoch', 'residual', 'p_norm', 'hr10_clean', 'test_auc_clean']].to_string(index=False))
