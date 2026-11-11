import os, glob, random, time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
import ifru_sweep as S

DATASET = os.environ.get('DATASET', 'MovieLens')
RATIO = os.environ.get('RATIO', '0.03')
NF = int(os.environ.get('N_FILLERS', 150))
START = os.environ.get('START', 'original')
EPOCHS = int(os.environ.get('EPOCHS', 150))
EVAL_EVERY = int(os.environ.get('EVAL_EVERY', 5))
FT_LR = float(os.environ.get('FT_LR', 1e-4))
TAUS = [None if x.strip() == 'none' else float(x) for x in os.environ.get('RESET_TAUS', 'none,0.9,0.5').split(',')]

cfg = dict(embed_size=32, batch_size=2048, epoch=200, data_path='Data/Process/',
           dataset=DATASET, attack=RATIO, k_hop=1, data_type='retraining', if_epoch=200,
           if_lr=1e4, if_init_std=0, seed=1024, lr=1e-4, regs=0, init_std=1e-4, n_fillers=NF)
args = S.model_hyparameters()
args.reset(cfg)

path = args.data_path + args.dataset + '/' + args.attack
gen = S.Data_for_LightGCN(args, path=path)
gen.set_train_mode('retraining')

W = './Weights/LightGCN/'
orig_path = (W + 'LightGCN_lr-%s-embed_size-%s-batch_size-%s-data_type-full-dataset-%s-attack-%s-'
             'seed-%s-init_std-%s-n_fillers-%s-m.pth'
             % (args.lr, args.embed_size, args.batch_size, DATASET, RATIO, args.seed, args.init_std, NF))
ifru_path = ('Weights/LightGCN_IFRU/lightgcn_dataset_%s_filler_%s_attack_%s_lr_%s_khop_1_emb_32_noES.pth'
             % (DATASET, NF, RATIO, float(os.environ.get('IF_LR', 1e4))))
start_path = None if START == 'scratch' else (orig_path if START == 'original' else ifru_path)
cand = glob.glob(W + '*data_type-retraining-dataset-%s-attack-%s-*n_fillers-%s-*.pth' % (DATASET, RATIO, NF))
print('시작 가중치:', start_path, START == 'scratch' or os.path.isfile(start_path))

model = S.LightGCN(args, dataset=gen).cuda()
tg, n_fake = S.infer_internal_target(args.data_path, args.dataset, args.attack)
targets = list(tg) if isinstance(tg, (list, tuple)) else [int(tg)]
ctx = S.build_hr_context(gen, targets, k=10)
test = gen.test[['user', 'item', 'label']].values
valid = gen.valid[['user', 'item', 'label']].values

npos = gen.train_normal[gen.train_normal['label'] > 0]
fpos = gen.train_random[gen.train_random['label'] > 0]
dn = np.bincount(npos['item'].values.astype(int), minlength=gen.n_items).astype(float)
df = np.bincount(fpos['item'].values.astype(int), minlength=gen.n_items).astype(float)
ratio = df / np.maximum(dn + df, 1e-9)


def evaluate():
    hr = S.compute_hr10(model, model.Graph, ctx)
    t = roc_auc_score(test[:, -1], model.predict(test[:, 0], test[:, 1]))
    v = roc_auc_score(valid[:, -1], model.predict(valid[:, 0], valid[:, 1]))
    return hr, v, t


ref_auc = None
if cand:
    model.load_state_dict(torch.load(sorted(cand)[0], map_location='cuda'))
    hr, v, t = evaluate()
    ref_auc = t
    print('기준: 전체 재학습   타깃 노출 %.4f%%  test AUC %.4f' % (100 * hr, t))

os.makedirs('results_reset', exist_ok=True)
summary = []
for tau in TAUS:
    tag = 'none' if tau is None else ('%g' % tau)
    random.seed(1024); np.random.seed(1024); torch.manual_seed(1024); torch.cuda.manual_seed_all(1024)
    if START == 'scratch':
        torch.nn.init.normal_(model.embedding_user.weight, std=args.init_std)
        torch.nn.init.normal_(model.embedding_item.weight, std=args.init_std)
    else:
        model.load_state_dict(torch.load(start_path, map_location='cuda'))
    n_reset = 0
    if tau is not None:
        sel = torch.tensor(np.where(ratio >= tau)[0]).long().cuda()
        model.embedding_item.weight.data[sel] = torch.randn(len(sel), args.embed_size).cuda() * 1e-4
        n_reset = len(sel)
    print()
    print('=== 시작 %s | 초기화 문턱 %s (아이템 %d개) | 추가 학습 %d epoch ===' % (START, tag, n_reset, EPOCHS))
    opt = torch.optim.Adam(model.parameters(), lr=FT_LR)
    loader = gen.generate_train_dataloader(batch_size=args.batch_size) if EPOCHS > 0 else None
    rows, spent = [], 0.0

    def snap(ep):
        hr, v, t = evaluate()
        rows.append(dict(start=START, tau=tag, epoch=ep, seconds=spent, hr10_clean=hr, valid_auc=v, test_auc=t))
        print('epoch %3d | %7.1f초 | 타깃 노출 %8.4f%% | valid %.4f | test %.4f' % (ep, spent, 100 * hr, v, t))

    snap(0)
    for ep in range(1, EPOCHS + 1):
        torch.cuda.synchronize(); t0 = time.time()
        for batch in loader:
            opt.zero_grad()
            bu, bi, bl = batch[:, 0].cuda().long(), batch[:, 1].cuda().long(), batch[:, -1].cuda().float()
            loss = model.compute_bce_loss(bu, bi, bl)
            loss.backward()
            opt.step()
        torch.cuda.synchronize(); spent += time.time() - t0
        if ep % EVAL_EVERY == 0 or ep == EPOCHS:
            snap(ep)

    d = pd.DataFrame(rows)
    d.to_csv('results_reset/ft_%s_%s_start-%s_tau-%s.csv' % (DATASET, RATIO, START, tag), index=False)
    best = d.loc[d['valid_auc'].idxmax()]
    line = '[요약] 시작 %s, 문턱 %s | 0 epoch: 노출 %.4f%% AUC %.4f' % (
        START, tag, 100 * d['hr10_clean'].iloc[0], d['test_auc'].iloc[0])
    if EPOCHS > 0:
        line += ' | valid 최고 epoch %d (%.0f초): 노출 %.4f%% AUC %.4f' % (
            best['epoch'], best['seconds'], 100 * best['hr10_clean'], best['test_auc'])
        if ref_auc is not None:
            hit = d[d['test_auc'] >= ref_auc]
            if len(hit):
                h = hit.iloc[0]
                line += ' | 재학습 AUC 도달 epoch %d (%.0f초, 노출 %.4f%%)' % (h['epoch'], h['seconds'], 100 * h['hr10_clean'])
            else:
                line += ' | 재학습 AUC 미도달'
    summary.append(line)

print()
for s in summary:
    print(s)
