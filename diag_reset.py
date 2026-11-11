import os, glob, random, time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import ifru_sweep as S

DATASET = os.environ.get('DATASET', 'MovieLens')
RATIO = os.environ.get('RATIO', '0.03')
NF = int(os.environ.get('N_FILLERS', 150))
REFIT_LR = float(os.environ.get('REFIT_LR', 0.01))

cfg = dict(embed_size=32, batch_size=2048, epoch=200, data_path='Data/Process/',
           dataset=DATASET, attack=RATIO, k_hop=1, data_type='full', if_epoch=200,
           if_lr=1e4, if_init_std=0, seed=1024, lr=1e-4, regs=0, init_std=1e-4, n_fillers=NF)
args = S.model_hyparameters()
args.reset(cfg)
random.seed(1024); np.random.seed(1024); torch.manual_seed(1024)

path = args.data_path + args.dataset + '/' + args.attack
gen = S.Data_for_LightGCN(args, path=path)
gen.set_train_mode('full')
clean = S.Data_for_LightGCN(args, path=path)
clean.set_train_mode('retraining')
clean_graph = clean.Graph

W = './Weights/LightGCN/'
orig_path = (W + 'LightGCN_lr-%s-embed_size-%s-batch_size-%s-data_type-full-dataset-%s-attack-%s-'
             'seed-%s-init_std-%s-n_fillers-%s-m.pth'
             % (args.lr, args.embed_size, args.batch_size, DATASET, RATIO, args.seed, args.init_std, NF))
cand = glob.glob(W + '*data_type-retraining-dataset-%s-attack-%s-*n_fillers-%s-*.pth' % (DATASET, RATIO, NF))

model = S.LightGCN(args, dataset=gen).cuda()
model.load_state_dict(torch.load(orig_path, map_location='cuda'))
U0 = model.embedding_user.weight.data.clone()
I0 = model.embedding_item.weight.data.clone()

tg, n_fake = S.infer_internal_target(args.data_path, args.dataset, args.attack)
targets = list(tg) if isinstance(tg, (list, tuple)) else [int(tg)]
ctx = S.build_hr_context(gen, targets, k=10)
test = gen.test[['user', 'item', 'label']].values

npos = gen.train_normal[gen.train_normal['label'] > 0]
fpos = gen.train_random[gen.train_random['label'] > 0]
dn = np.bincount(npos['item'].values.astype(int), minlength=gen.n_items).astype(float)
df = np.bincount(fpos['item'].values.astype(int), minlength=gen.n_items).astype(float)
ratio = df / np.maximum(dn + df, 1e-9)
ratio_t = torch.tensor(ratio).float().cuda().unsqueeze(1)

print()
print('데이터셋 %s, 공격 %s, 타깃 %d개' % (DATASET, RATIO, len(targets)))
print('타깃의 가짜 비율:', ['%.3f' % ratio[t] for t in targets])
for tau in (0.5, 0.7, 0.9, 0.95, 0.99):
    print('가짜 비율 %.2f 이상 아이템: %d개' % (tau, int((ratio >= tau).sum())))


def put(u, i):
    model.embedding_user.weight.data.copy_(u)
    model.embedding_item.weight.data.copy_(i)


def ev(tag):
    og = model.Graph
    model.Graph = clean_graph
    hr = S.compute_hr10(model, clean_graph, ctx)
    auc = roc_auc_score(test[:, -1], model.predict(test[:, 0], test[:, 1]))
    model.Graph = og
    print('%-52s 타깃 노출 %8.4f%%   AUC %.4f' % (tag, 100 * hr, auc))


print()
print('=== 1. 기준 ===')
put(U0, I0); ev('Graph-only')
if cand:
    sd = torch.load(sorted(cand)[0], map_location='cuda')
    put(sd['embedding_user.weight'], sd['embedding_item.weight']); ev('전체 재학습')

print()
print('=== 2. 문턱 이상 아이템을 0으로 (학습 없음) ===')
for tau in (0.5, 0.7, 0.9, 0.95, 0.99):
    i = I0.clone()
    i[torch.tensor(np.where(ratio >= tau)[0]).long().cuda()] = 0
    put(U0, i); ev('가짜 비율 >= %.2f 초기화' % tau)

print()
print('=== 3. 남은 지지 비율만큼 축소: e <- (1 - 가짜비율) * e ===')
put(U0, I0 * (1 - ratio_t)); ev('전 아이템 축소')
i = I0.clone(); m = (ratio_t >= 0.5).float()
put(U0, i * (1 - ratio_t * m)); ev('가짜 비율 0.5 이상만 축소')

print()
print('=== 4. 초기화 후 그 아이템만 정상 데이터로 다시 맞춤 (나머지 고정, lr %g) ===' % REFIT_LR)


def refit(tau, steps=(50, 100, 200, 400)):
    idx = np.where(ratio >= tau)[0]
    rows = gen.train_normal[gen.train_normal['item'].isin(idx)][['user', 'item', 'label']].values
    put(U0, I0)
    sel = torch.tensor(idx).long().cuda()
    torch.manual_seed(1024)
    model.embedding_item.weight.data[sel] = torch.randn(len(sel), I0.shape[1]).cuda() * 1e-4
    if len(rows) == 0:
        ev('tau %.2f: 남은 정상 데이터 없음' % tau)
        return
    u = torch.tensor(rows[:, 0].astype(np.int64)).cuda()
    it = torch.tensor(rows[:, 1].astype(np.int64)).cuda()
    y = torch.tensor(rows[:, 2].astype(np.float32)).cuda()
    mask = torch.zeros(I0.shape[0], 1).cuda()
    mask[sel] = 1
    opt = torch.optim.Adam([model.embedding_item.weight], lr=REFIT_LR)
    og = model.Graph
    spent = 0.0
    for s in range(1, max(steps) + 1):
        model.Graph = clean_graph
        torch.cuda.synchronize(); t0 = time.time()
        model.zero_grad()
        loss = model.compute_bce_loss(u, it, y)
        loss.backward()
        model.embedding_item.weight.grad.mul_(mask)
        opt.step()
        torch.cuda.synchronize(); spent += time.time() - t0
        model.Graph = og
        if s in steps:
            ev('tau %.2f | 아이템 %d개, 정상 %d행 | %3d스텝 %.1f초' % (tau, len(idx), len(rows), s, spent))
    model.zero_grad()


for tau in (0.5, 0.9, 0.99):
    refit(tau)
