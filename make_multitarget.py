import os
import sys

EDITS_IFRU = [
    ("""    if len(candidates) != 1:
        raise RuntimeError(
            '타깃 아이템을 특정할 수 없음. ratio=%s, 가짜 유저=%d, 후보=%s\\n'
            '이 상태로 HR@10을 측정하면 안 된다.'
            % (ratio, fake_user_count, candidates)
        )
    return candidates[0], int(fake_user_count)""",
     """    expected = os.environ.get('N_TARGETS')
    if not candidates:
        raise RuntimeError(
            '타깃 아이템을 찾을 수 없음. ratio=%s, 가짜 유저=%d'
            % (ratio, fake_user_count))
    if expected and len(candidates) != int(expected):
        raise RuntimeError(
            '타깃 개수가 N_TARGETS(%s)와 다름. ratio=%s, 찾은 개수=%d, 후보=%s\\n'
            '이 상태로 HR@10을 측정하면 안 된다.'
            % (expected, ratio, len(candidates), candidates))
    return candidates, int(fake_user_count)"""),
    ("def build_hr_context(data_generator, target_item, k=10):",
     "def build_hr_context(data_generator, target_items, k=10):"),
    ("""    return {
        'users': torch.from_numpy(eval_users).long().cuda(),
        'seen': seen.cuda(),
        'target': int(target_item),
        'k': k,
        'n_eval': int(len(eval_users)),
    }""",
     """    if isinstance(target_items, (list, tuple, set, np.ndarray)):
        _t = sorted(int(x) for x in target_items)
    else:
        _t = [int(target_items)]

    return {
        'users': torch.from_numpy(eval_users).long().cuda(),
        'seen': seen.cuda(),
        'targets': torch.tensor(_t, dtype=torch.long).cuda(),
        'target_list': _t,
        'target': _t[0],
        'k': k,
        'n_eval': int(len(eval_users)),
    }"""),
    ("""        topk = scores.topk(ctx['k'], dim=1).indices
        hit = (topk == ctx['target']).any(dim=1).float().mean().item()
        del scores, topk, all_u, all_i
    return float(hit)""",
     """        topk = scores.topk(ctx['k'], dim=1).indices
        tg = ctx['targets'].view(1, 1, -1)
        per_user = (topk.unsqueeze(-1) == tg).any(dim=1).float()   # [유저, 타깃]
        per_target = per_user.mean(dim=0)                          # 타깃별 HR@10
        hit = float(per_target.mean().item())                      # 타깃 평균
        ctx['last_per_target'] = [float(v) for v in per_target.detach().cpu().numpy()]
        del scores, topk, all_u, all_i, per_user, per_target
    return float(hit)"""),
    ("""    print('타깃 아이템 내부 ID: %d (가짜 유저 %d명)' % (target_item, fake_user_count))""",
     """    print('타깃 아이템 내부 ID: %s (%d개, 가짜 유저 %d명)'
          % (target_item, len(target_item), fake_user_count))"""),
    ("        'target_item': target_item,",
     "        'target_item': ' '.join(str(t) for t in hr_ctx['target_list']),"),
    ("        config_args['dataset'] = 'MovieLens'",
     "        config_args['dataset'] = os.environ.get('DATASET', 'MovieLens')"),
]

EDITS_TRAIN = [
    ("ATTACK_RATIOS = ('0.01', '0.03', '0.05')",
     """ATTACK_RATIOS = ('0.01', '0.03', '0.05')
_r = os.environ.get('RATIOS')
if _r:
    ATTACK_RATIOS = tuple(x.strip() for x in _r.split(',') if x.strip())"""),
    ("            'dataset': 'MovieLens',",
     "            'dataset': os.environ.get('DATASET', 'MovieLens'),"),
]


def apply(src, dst, edits, label):
    if not os.path.exists(src):
        sys.exit('!! %s 가 없습니다.' % src)
    s = open(src, encoding='utf-8').read()
    for n, (old, new) in enumerate(edits, 1):
        if new.split('\n')[0] in s and old not in s:
            print('   [%s %d] 이미 적용됨, 건너뜀' % (label, n))
            continue
        if s.count(old) != 1:
            sys.exit('!! [%s %d] 기준 문구가 %d번 나타남:\n%s' % (label, n, s.count(old), old[:90]))
        s = s.replace(old, new, 1)
        print('   [%s %d] 적용' % (label, n))
    open(dst, 'w', encoding='utf-8').write(s)
    print('>> %s 저장' % dst)


print('[1/2] ifru_sweep.py')
apply('ifru_sweep.py', 'ifru_sweep.py', EDITS_IFRU, 'ifru')
print('[2/2] train_original_ds.py')
apply('original_lightgcn_movie_server_final.py', 'train_original_ds.py', EDITS_TRAIN, 'train')
