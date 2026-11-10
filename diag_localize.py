import os, glob, random
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
import ifru_sweep as S

DATASET = os.environ.get('DATASET', 'MovieLens')
RATIO = os.environ.get('RATIO', '0.03')
NF = int(os.environ.get('N_FILLERS', 150))

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
print('Original :', orig_path, os.path.isfile(orig_path))
print('Retrain  :', cand)

model = S.LightGCN(args, dataset=gen).cuda()
model.load_state_dict(torch.load(orig_path, map_location='cuda'))
U0 = model.embedding_user.weight.data.clone()
I0 = model.embedding_item.weight.data.clone()
UR = IR = None
if cand:
    sd = torch.load(sorted(cand)[0], map_location='cuda')
    UR = sd['embedding_user.weight'].clone()
    IR = sd['embedding_item.weight'].clone()

tg, n_fake = S.infer_internal_target(args.data_path, args.dataset, args.attack)
targets = list(tg) if isinstance(tg, (list, tuple)) else [int(tg)]
ctx = S.build_hr_context(gen, targets, k=10)
test = gen.test[['user', 'item', 'label']].values
T = torch.tensor(targets).long().cuda()

npos = gen.train_normal[gen.train_normal['label'] > 0]
fpos = gen.train_random[gen.train_random['label'] > 0]
dn = np.bincount(npos['item'].values.astype(int), minlength=gen.n_items).astype(float)
df = np.bincount(fpos['item'].values.astype(int), minlength=gen.n_items).astype(float)
ratio = df / np.maximum(dn + df, 1e-9)
heavy = torch.tensor(np.where(ratio >= 0.5)[0]).long().cuda()
fake_users = torch.tensor(np.unique(gen.train_random['user'].values.astype(int))).long().cuda()

print()
print('타깃 %s | 가짜 유저 %d명' % (targets, n_fake))
print('타깃의 정상 긍정 평점 수:', [int(dn[t]) for t in targets])
print('타깃의 가짜 긍정 평점 수:', [int(df[t]) for t in targets])
print('가짜 비율 0.5 이상 아이템: %d개' % len(heavy))


def put(u, i):
    model.embedding_user.weight.data.copy_(u)
    model.embedding_item.weight.data.copy_(i)


def ev(tag):
    og = model.Graph
    model.Graph = clean_graph
    hr = S.compute_hr10(model, clean_graph, ctx)
    auc = roc_auc_score(test[:, -1], model.predict(test[:, 0], test[:, 1]))
    model.Graph = og
    print('%-46s 타깃 노출 %8.4f%%   AUC %.4f' % (tag, 100 * hr, auc))


print()
print('=== clean 그래프에서 평가 ===')
put(U0, I0); ev('A. Original 가중치 그대로 (Graph-only)')

i = I0.clone(); i[T] = 0
put(U0, i); ev('B. 타깃 임베딩만 0으로')

i = I0.clone(); i[T] = torch.randn(len(T), I0.shape[1]).cuda() * 1e-4
put(U0, i); ev('C. 타깃 임베딩만 초기값 분포로')

i = I0.clone(); i[heavy] = 0
put(U0, i); ev('D. 가짜 비율 0.5 이상 아이템 전부 0으로')

if IR is not None:
    i = I0.clone(); i[T] = IR[T]
    put(U0, i); ev('E. 타깃 임베딩만 재학습 것으로 교체')
    i = I0.clone(); i[heavy] = IR[heavy]
    put(U0, i); ev('F. 가짜 비율 0.5 이상 아이템을 재학습 것으로')
    put(U0, IR); ev('G. 아이템 전부 재학습, 유저는 Original')
    put(UR, I0); ev('H. 유저 전부 재학습, 아이템은 Original')
    put(UR, IR); ev('I. 재학습 모델 (기준)')

print()
print('=== 타깃 임베딩의 기하 ===')
med = I0.norm(dim=1).median().item()
fu = U0[fake_users].mean(dim=0)
for k, t in enumerate(targets):
    line = '타깃 %d | 노름 Original %.4f (아이템 중앙값 %.4f)' % (t, I0[t].norm().item(), med)
    line += ' | 가짜유저 평균방향과 cos %.3f' % torch.cosine_similarity(I0[t], fu, dim=0).item()
    if IR is not None:
        line += ' | 재학습 노름 %.4f, cos(Orig,재학습) %.3f' % (
            IR[t].norm().item(), torch.cosine_similarity(I0[t], IR[t], dim=0).item())
    print(line)

print()
print('=== 가짜 데이터가 얼마나 확신 있게 학습됐나 (full 그래프, Original) ===')
put(U0, I0)
fr = fpos[['user', 'item']].values
pred = model.predict(fr[:, 0], fr[:, 1])
print('가짜 긍정 %d행의 예측 확률: 평균 %.4f, 중앙값 %.4f, 0.9 초과 비율 %.1f%%'
      % (len(pred), pred.mean(), np.median(pred), 100 * (pred > 0.9).mean()))
m = np.isin(fr[:, 1], targets)
print('그중 타깃 행 %d개: 평균 %.4f  (기울기 크기는 1 - 예측확률에 비례)' % (m.sum(), pred[m].mean()))
