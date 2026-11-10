import os
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from utility.load_data import * 
import scipy.sparse as sp
import torch.nn.functional as F
from Model.MF import MF
import time
from sklearn.metrics import roc_auc_score
import time
from torch.autograd import Variable
from Model.Lightgcn import LightGCN
from utility.compute import *


ATTACK_RATIOS = ('0.01', '0.03', '0.05')

RESULT_DIR = 'results'


# ----------------------------------------------------------------------------
# 계측용 유틸
#
# 1) clean 그래프 평가  - 가짜 간선을 뺀 그래프에서도 같이 측정한다.
# 2) epoch별 HR@10     - 타깃 아이템이 Top-10에 남아 있는 유저 비율.
# 3) p 수렴 확인        - |p|와 잔차 |H·p - g|를 매 epoch 기록한다.
# 4) 소요 시간          - 보정 연산만 누적한다. 평가/로그 시간은 제외.
#                        GPU 연산이 비동기라 앞뒤로 synchronize를 건다.
# ----------------------------------------------------------------------------


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def infer_internal_target(data_path, dataset, ratio):
    """train_random.csv에서 타깃 아이템의 내부 ID를 찾는다.

    공격 비율마다 데이터셋이 새로 구성되므로 내부 ID가 달라진다.
    주입된 타깃은 '모든 가짜 유저가 평가한 아이템' 하나뿐이다.
    generate_recommendation_scores_final.py와 같은 방식이다.
    """
    path = os.path.join(data_path, dataset, ratio, 'train_random.csv')
    attack = pd.read_csv(path)
    fake_user_count = attack['user'].nunique()
    if fake_user_count == 0:
        raise ValueError('가짜 유저가 없음: %s' % path)

    positive = attack[attack['label'] > 0][['user', 'item']].drop_duplicates()
    coverage = positive.groupby('item')['user'].nunique()
    candidates = sorted(
        int(item) for item, cnt in coverage.items()
        if int(cnt) == int(fake_user_count)
    )
    if len(candidates) != 1:
        raise RuntimeError(
            '타깃 아이템을 특정할 수 없음. ratio=%s, 가짜 유저=%d, 후보=%s\n'
            '이 상태로 HR@10을 측정하면 안 된다.'
            % (ratio, fake_user_count, candidates)
        )
    return candidates[0], int(fake_user_count)


def build_hr_context(data_generator, target_item, k=10):
    """HR@10 측정에 필요한 것을 한 번만 만들어 둔다.

    평가 유저는 test에 등장하는 유저이고, 후보 아이템에서는
    그 유저가 이미 긍정적으로 상호작용한 아이템을 제외한다.
    (generate_recommendation_scores_final.py와 동일한 규칙)
    """
    eval_users = np.sort(data_generator.test['user'].unique().astype(np.int64))
    n_items = int(data_generator.n_items)

    pos = data_generator.train[data_generator.train['label'] > 0]
    pos = pos[['user', 'item']].values.astype(np.int64)

    seen = torch.zeros((len(eval_users), n_items), dtype=torch.bool)
    if len(pos):
        rows = np.searchsorted(eval_users, pos[:, 0])
        rows_clipped = np.clip(rows, 0, len(eval_users) - 1)
        keep = (eval_users[rows_clipped] == pos[:, 0]) & (pos[:, 1] < n_items)
        seen[rows_clipped[keep], pos[keep, 1]] = True

    return {
        'users': torch.from_numpy(eval_users).long().cuda(),
        'seen': seen.cuda(),
        'target': int(target_item),
        'k': k,
        'n_eval': int(len(eval_users)),
    }


def compute_hr10(model, graph, ctx):
    """주어진 그래프에서 타깃 아이템의 HR@10을 구한다."""
    with torch.no_grad():
        all_u, all_i = model.F_computer(
            model.embedding_user.weight, model.embedding_item.weight, graph)
        scores = all_u[ctx['users']] @ all_i.t()
        scores.masked_fill_(ctx['seen'], float('-inf'))
        topk = scores.topk(ctx['k'], dim=1).indices
        hit = (topk == ctx['target']).any(dim=1).float().mean().item()
        del scores, topk, all_u, all_i
    return float(hit)


class model_hyparameters(object):
    def __init__(self):
        super().__init__()
        self.embed_size = 48
        self.batch_size = 2048
        self.epoch = 200
        self.data_path = 'Data/Process/'
        self.dataset = 'MovieLens'
        self.attack = '0.01'
        self.k_hop = 0
        self.data_type = 'full'
        self.if_epoch = 200
        self.if_lr = 1e4
        self.if_init_std = 0
        self.seed = 1024
        self.lr = 1e-3
        self.regs = 0
        self.init_std = 0
        self.n_fillers = 100

        # lightgcn hyper-parameters
        self.gcn_layers = 1
        self.keep_prob = 1
        self.A_n_fold = 10
        self.A_split = False
        self.dropout = False
        self.pretrain = 0
        
    def reset(self, config):
        for name,val in config.items():
            setattr(self,name,val)


class influence_unlearn(nn.Module):
    def __init__(self,save_name,if_epoch=100,if_lr=1e-2,k_hop=1,init_range=1e-4) -> None:
        super(influence_unlearn).__init__()
        self.if_epoch = if_epoch
        self.if_lr = if_lr
        self.k_hop = k_hop
        self.range = init_range
        self.save_name = save_name
        self.p = None

    def compute_hessian_with_test(self, model=None, data_generator=None,
                                  clean_graph=None, hr_ctx=None, trace_path=None):
        # 소요 시간 측정 시작. 가중치는 이미 로드된 상태에서 호출된다.
        self.compute_time = 0.0
        _sync()
        _t_setup = time.time()

        nei_users, nei_items = compute_neighbor(data_generator, 0)
        nei_users = torch.from_numpy(nei_users).cuda().long()
        nei_items = torch.from_numpy(nei_items).cuda().long()

        mask = get_eval_mask(data_generator)

        _sync()
        self.compute_time += time.time() - _t_setup

        # ---- 아래는 계측용 출력. 시간에 포함하지 않는다 ----
        valid_auc, valid_auc_or, valid_auc_and, test_auc, test_auc_or, test_auc_and = get_eval_result(data_generator, model, mask)
        print("results before unlearning: valid auc:%.6f, valid auc or:%.6f, valid auc and:%.6f" %(valid_auc, valid_auc_or, valid_auc_and))
        print("results before unlearning: test auc:%.6f, test auc or:%.6f, test auc and:%.6f" %(test_auc, test_auc_or, test_auc_and))
        if hr_ctx is not None:
            print("results before unlearning: HR@10 full:%.4f%%" % (100.0 * compute_hr10(model, model.Graph, hr_ctx)))
            if clean_graph is not None:
                print("results before unlearning: HR@10 clean:%.4f%%" % (100.0 * compute_hr10(model, clean_graph, hr_ctx)))

        _sync()
        _t_setup = time.time()

        nei_users, nei_items = self.compute_neighbor_influence_clip(data_generator, k_hop=self.k_hop)
        nei_users = torch.from_numpy(nei_users).cuda().long()
        nei_items = torch.from_numpy(nei_items).cuda().long()
        un_u_para = model.embedding_user.weight[nei_users].reshape(-1)
        un_i_para = model.embedding_item.weight[nei_items].reshape(-1)
        u_para_num = un_u_para.shape[0]
        i_para_num = un_i_para.shape[0]

        un_ui_para = torch.cat([un_u_para,un_i_para],dim=-1)
        u_para = model.embedding_user.weight.clone().detach()
        i_para = model.embedding_item.weight.clone().detach()
        u_para[nei_users] = un_ui_para[:u_para_num].reshape(-1, u_para.shape[-1])
        i_para[nei_items] = un_ui_para[u_para_num:].reshape(-1, i_para.shape[-1])

        def loss_fun(para1, para2):
            u_para,i_para = para1,para2
            learned_data = data_generator.train[['user','item','label']].values
            learned_data = torch.from_numpy(learned_data).cuda()
            user_embs = u_para[learned_data[:,0].long()]
            item_embs = i_para[learned_data[:,1].long()]
            scores = torch.mul(user_embs, item_embs).sum(dim=-1)
            bce_loss = F.binary_cross_entropy_with_logits(scores, learned_data[:,-1].float(), reduction='mean')
            return bce_loss
        
        def unlearn_loss_fun(para1,para2):
            u_para, i_para = para1, para2
            rm_data = data_generator.train_random.values
            rm_data = torch.from_numpy(rm_data).cuda()
            adj_graph = model.Graph
            user_emb_gnn, item_emb_gnn = model.F_computer(u_para, i_para, adj_graph)
            user_embs_rm = user_emb_gnn[rm_data[:,0].long()]
            item_embs_rm = item_emb_gnn[rm_data[:,1].long()]
            scores = torch.mul(user_embs_rm, item_embs_rm).sum(dim=-1)
            rm_loss = F.binary_cross_entropy_with_logits(scores, rm_data[:,2].float(), reduction='sum')
            
            # the loss of changed data
            ### original loss
            ch_data = data_generator.changed_data.values[:,0:3]
            ch_data = torch.from_numpy(ch_data).cuda()
            user_emb_ch1 = user_emb_gnn[ch_data[:,0].long()]
            item_emb_ch1 = item_emb_gnn[ch_data[:,1].long()]
            ch_score1 = torch.mul(user_emb_ch1,item_emb_ch1).sum(dim=-1)
            ch_loss_1 = F.binary_cross_entropy_with_logits(ch_score1, ch_data[:,2].float(), reduction='sum')
            ### changed loss
            ch_graph = data_generator.ChangedGraph
            user_emb_gnn, item_emb_gnn = model.F_computer(u_para, i_para, ch_graph)
            user_emb_ch2 = user_emb_gnn[ch_data[:,0].long()]
            item_emb_ch2 = item_emb_gnn[ch_data[:,1].long()]
            ch_score_2 = torch.mul(user_emb_ch2,item_emb_ch2).sum(dim=-1)
            ch_loss_2 = F.binary_cross_entropy_with_logits(ch_score_2, ch_data[:,2].float(), reduction='sum')
            ### loss for changed data
            ch_loss = ch_loss_1 - ch_loss_2
            ### total unlearn loss
            unlearn_loss = rm_loss + ch_loss
            return unlearn_loss

        total_loss = loss_fun(u_para, i_para)
        total_grad = torch.autograd.grad(total_loss, un_ui_para, create_graph=True, retain_graph=True)[0].reshape(-1,1)
        unlearn_loss = unlearn_loss_fun(u_para, i_para)
        unlearn_grad = torch.autograd.grad(unlearn_loss, un_ui_para,retain_graph=True)[0].reshape(-1,1)

        def hvp(grad, vec):
            vec = vec.detach()
            prod = torch.mul(vec, grad).sum()
            res = torch.autograd.grad(prod, un_ui_para, retain_graph=True)[0]
            return res.detach()
        def grad_goal(grad, vec):
            return hvp(grad, vec).unsqueeze(-1) - unlearn_grad.detach()
        
        self.p = Variable(torch.empty([unlearn_grad.shape[0],1])).cuda()
        nn.init.uniform_(self.p, -self.range, self.range)
        opt = torch.optim.Adam([self.p], lr=self.if_lr)

        best_auc = 0
        best_epoch = 0
        best_test_auc = 0
        best_nei_auc = 0
        not_change = 0

        _sync()
        self.compute_time += time.time() - _t_setup
        setup_time = self.compute_time
        print("[time] 루프 진입 전 준비(기울기·헤시안 재료 계산): %.3f초" % setup_time)

        t0 = time.time()
        trace_rows = []

        for if_ep in range(self.if_epoch):
            # ---- 여기부터 보정 연산. 시간에 포함 ----
            _sync()
            s_time = time.time()

            opt.zero_grad()
            self.p.grad = grad_goal(total_grad, self.p)
            opt.step()
            with torch.no_grad():
                un_ui_para_temp = un_ui_para + 1.0/data_generator.n_train * self.p.squeeze()
                model.embedding_user.weight.data[nei_users] = un_ui_para_temp[:u_para_num].reshape(-1, u_para.shape[-1]).data + 0
                model.embedding_item.weight.data[nei_items] = un_ui_para_temp[u_para_num:].reshape(-1, i_para.shape[-1]).data + 0

            _sync()
            e_time = time.time()
            step_time = e_time - s_time
            self.compute_time += step_time
            # ---- 보정 연산 끝. 아래는 계측이므로 시간에 포함하지 않는다 ----

            with torch.no_grad():
                # p 수렴 확인. self.p.grad가 곧 잔차 (H·p - g)다.
                p_norm = float(torch.norm(self.p).item())
                residual = float(torch.norm(self.p.grad).item())

                valid_auc, valid_auc_or, valid_auc_and, test_auc, test_auc_or, test_auc_and = get_eval_result(data_generator, model, mask)

                hr_full = float('nan')
                hr_clean = float('nan')
                v_clean = float('nan')
                t_clean = float('nan')

                if hr_ctx is not None:
                    hr_full = compute_hr10(model, model.Graph, hr_ctx)
                    if clean_graph is not None:
                        hr_clean = compute_hr10(model, clean_graph, hr_ctx)

                if clean_graph is not None:
                    _orig_graph = model.Graph
                    model.Graph = clean_graph
                    v_clean, _, _, t_clean, _, _ = get_eval_result(data_generator, model, mask)
                    model.Graph = _orig_graph

                print("epoch: %d, step: %.6f | full valid:%.6f test:%.6f | clean valid:%.6f test:%.6f "
                      "| HR@10 full:%.4f%% clean:%.4f%% | |p|:%.4e resid:%.4e"
                      % (if_ep, step_time, valid_auc, test_auc, v_clean, t_clean,
                         100.0*hr_full, 100.0*hr_clean, p_norm, residual))

                trace_rows.append({
                    'epoch': if_ep,
                    'step_time': step_time,
                    'cumulative_time': self.compute_time,
                    'valid_auc_full': valid_auc,
                    'test_auc_full': test_auc,
                    'valid_auc_clean': v_clean,
                    'test_auc_clean': t_clean,
                    'hr10_full': hr_full,
                    'hr10_clean': hr_clean,
                    'p_norm': p_norm,
                    'residual': residual,
                })

                if valid_auc > best_auc:
                    best_auc = valid_auc
                    best_epoch = if_ep
                    best_test_auc = test_auc

                torch.save(model.state_dict(), self.save_name)

        print("AUC 기준 best epoch는 %d였으나, 최종 epoch %d 모델을 저장함" % (best_epoch, if_ep))
        print("[time] 언러닝 연산 시간(준비 + 보정, 평가 제외): %.3f초" % self.compute_time)
        print("[time] 벽시계 전체(평가·저장 포함): %.3f초" % (time.time() - t0 + setup_time))

        if trace_path is not None and len(trace_rows):
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            pd.DataFrame(trace_rows).to_csv(trace_path, index=False)
            print("[trace] epoch별 기록 저장:", trace_path)

        

    def compute_neighbor_influence_clip(self, data_generator, k_hop=1):
        train_data = data_generator.train.values.copy()
        matrix_size = data_generator.n_users + data_generator.n_items
        train_data[:,1] += data_generator.n_users
        train_data[:,-1] = np.ones_like(train_data[:,-1])

        train_data2 = np.ones_like(train_data)
        train_data2[:,0] = train_data[:,1]
        train_data2[:,1] = train_data[:,0]

        paddding = np.concatenate([np.arange(matrix_size).reshape(-1,1), np.arange(matrix_size).reshape(-1,1), np.ones(matrix_size).reshape(-1,1)],axis=-1)
        data = np.concatenate([train_data, train_data2, paddding],axis=0).astype(int)
        train_matrix = sp.csc_matrix((data[:,-1],(data[:,0],data[:,1])),shape=(matrix_size,matrix_size))

        degree = np.array(train_matrix.sum(axis=-1)).squeeze()
        
        unlearn_user = data_generator.train_random['user'].values.reshape(-1)
        unlearn_user, cunt_u = np.unique(unlearn_user, return_counts=True)
        unlearn_item = data_generator.train_random['item'].values.reshape(-1) + data_generator.n_users
        unlearn_item, cunt_i = np.unique(unlearn_item,return_counts=True)

        unlearn_ui = np.concatenate([unlearn_user, unlearn_item], axis=-1)
        unlearn_ui_cunt = np.concatenate([cunt_u, cunt_i], axis=-1)
        degree_k = degree[unlearn_ui]
        neighbor_set = dict(zip(unlearn_ui, unlearn_ui_cunt*1.0/degree_k))
        neighbor_set_list = [neighbor_set]
        pre_neighbor_set = neighbor_set
        print("neighbor_set size:", len(neighbor_set))

        for i in range(k_hop):
            neighbor_set = dict()
            existing_node = list(pre_neighbor_set.keys())
            nonzero_raw, nonzero_col = train_matrix[existing_node].nonzero()
            for kk in range(nonzero_raw.shape[0]):
                out_node =  existing_node[nonzero_raw[kk]]
                in_node = nonzero_col[kk]
                try:
                    neighbor_set[in_node] += pre_neighbor_set[out_node] *1.0/degree[in_node]
                except:
                    neighbor_set[in_node] = pre_neighbor_set[out_node] *1.0/degree[in_node]
            pre_neighbor_set = neighbor_set
            neighbor_set_list.append(neighbor_set)
        
        nei_dict = neighbor_set_list[1].copy()

        nei_weights = np.array(list(nei_dict.values()))
        nei_nodes = np.array(list(nei_dict.keys()))
        quantile_info = [np.quantile(nei_weights,m*0.1) for m in range(1,11)]
        print("quantile information (median 0.1-0.2--->1): ", quantile_info)
        
        select_index = np.where(nei_weights>quantile_info[3])
        neighbor_set = nei_nodes[select_index]
        print("neighbors before filtering:",nei_nodes.shape,"after filtering:", neighbor_set.shape)

        all_nei_ui = np.concatenate([unlearn_ui.squeeze(),neighbor_set.squeeze()])
        all_nei_ui = neighbor_set.squeeze()
        all_nei_ui = np.unique(all_nei_ui)
        print("total influenced users+items:",all_nei_ui.shape)
        return all_nei_ui[np.where(all_nei_ui<data_generator.n_users)], all_nei_ui[np.where(all_nei_ui>=data_generator.n_users)] - data_generator.n_users


def main(config_args):
    args = model_hyparameters()
    assert config_args is not None
    args.reset(config_args)

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)
    
    print('\n' + '=' * 70)
    print(f'[IFRU / unlearning] attack ratio: {args.attack}')
    print('=' * 70)

    data_generator = Data_for_LightGCN(args, path=args.data_path + args.dataset + '/' + args.attack)
    data_generator.set_train_mode(args.data_type)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    original_weight_path = (
        './Weights/LightGCN/'
        f'LightGCN_lr-{args.lr}-'
        f'embed_size-{args.embed_size}-'
        f'batch_size-{args.batch_size}-'
        f'data_type-full-'
        f'dataset-{args.dataset}-'
        f'attack-{args.attack}-'
        f'seed-{args.seed}-'
        f'init_std-{args.init_std}-'
        f'n_fillers-{args.n_fillers}-'
        'm.pth'
    )

    if not os.path.isfile(original_weight_path):
        raise FileNotFoundError(
            f'Original contaminated model not found: {original_weight_path}\n'
            'Run original_lightgcn_movie_server.py first.'
        )

    model = LightGCN(args, dataset=data_generator).cuda()
    print(f'Loading contaminated ORIGINAL model: {original_weight_path}')
    model.load_state_dict(torch.load(original_weight_path, map_location='cuda'))

    test_data = data_generator.test[['user','item','label']].values
    test_predictions =  model.predict(test_data[:,0], test_data[:,1])
    test_auc = roc_auc_score(test_data[:,-1],test_predictions)

    valid_data = data_generator.valid[['user','item','label']].values
    valid_predictions =  model.predict(valid_data[:,0], valid_data[:,1])
    valid_auc = roc_auc_score(valid_data[:,-1],valid_predictions)

    print("***************before unlearning*************")
    print("valid auc:",valid_auc)
    print("test auc:",test_auc)
    print(config_args)

    # ------------------------------------------------------------------
    # clean 그래프 준비
    #
    # 기존 설정은 IFRU를 full 그래프(가짜 간선 잔존)에서만 평가했다.
    # Retrain과 Fine-tuning은 clean 그래프에서 평가하므로 조건이 어긋난다.
    # 여기서 clean 그래프를 따로 만들어 두 조건을 모두 기록한다.
    #
    # getSparseGraph_mode_a는 self.Graph가 None일 때만 만들므로
    # 같은 객체에 set_train_mode를 두 번 불러도 다시 만들어지지 않는다.
    # 따라서 별도 인스턴스가 필요하다.
    # ------------------------------------------------------------------
    print('clean 그래프 구성 중 (가짜 유저 간선 제외)')
    clean_generator = Data_for_LightGCN(args, path=args.data_path + args.dataset + '/' + args.attack)
    clean_generator.set_train_mode('retraining')
    clean_graph = clean_generator.Graph

    target_item, fake_user_count = infer_internal_target(args.data_path, args.dataset, args.attack)
    print('타깃 아이템 내부 ID: %d (가짜 유저 %d명)' % (target_item, fake_user_count))

    hr_ctx = build_hr_context(data_generator, target_item, k=10)
    print('평가 유저 수: %d' % hr_ctx['n_eval'])

    os.makedirs("Weights/LightGCN_IFRU", exist_ok=True)
    ifru_save_name = "Weights/LightGCN_IFRU/lightgcn_dataset_{}_filler_{}_attack_{}_lr_{}_khop_{}_emb_{}_noES.pth".format(args.dataset, args.n_fillers, args.attack, args.if_lr, args.k_hop, args.embed_size)
    trace_path = os.path.join(
        RESULT_DIR,
        'ifru_trace_filler{}_attack{}.csv'.format(args.n_fillers, args.attack))

    unlearn = influence_unlearn(save_name=ifru_save_name,if_epoch=args.if_epoch, if_lr=args.if_lr, k_hop=args.k_hop, init_range=args.if_init_std)
    unlearn.compute_hessian_with_test(
        model=model,
        data_generator=data_generator,
        clean_graph=clean_graph,
        hr_ctx=hr_ctx,
        trace_path=trace_path,
    )
    model.load_state_dict(torch.load(ifru_save_name, map_location='cuda'))

    test_predictions =  model.predict(test_data[:,0], test_data[:,1])
    test_auc = roc_auc_score(test_data[:,-1],test_predictions)
    valid_predictions =  model.predict(valid_data[:,0], valid_data[:,1])
    valid_auc = roc_auc_score(valid_data[:,-1],valid_predictions)
    hr_full = compute_hr10(model, model.Graph, hr_ctx)
    print("***************after unlearning***************")
    print("[full 그래프]  valid auc: %.6f, test auc: %.6f" % (valid_auc, test_auc))
    print("[full 그래프]  HR@10: %.4f%%" % (100.0 * hr_full))

    _orig_graph = model.Graph
    model.Graph = clean_graph
    test_auc_c = roc_auc_score(test_data[:,-1], model.predict(test_data[:,0], test_data[:,1]))
    valid_auc_c = roc_auc_score(valid_data[:,-1], model.predict(valid_data[:,0], valid_data[:,1]))
    hr_clean = compute_hr10(model, clean_graph, hr_ctx)
    model.Graph = _orig_graph
    print("[clean 그래프] valid auc: %.6f, test auc: %.6f" % (valid_auc_c, test_auc_c))
    print("[clean 그래프] HR@10: %.4f%%" % (100.0 * hr_clean))

    summary = {
        'attack_ratio': args.attack,
        'n_fillers': args.n_fillers,
        'if_epoch': args.if_epoch,
        'if_lr': args.if_lr,
        'target_item': target_item,
        'eval_users': hr_ctx['n_eval'],
        'test_auc_full': test_auc,
        'test_auc_clean': test_auc_c,
        'hr10_full': 100.0 * hr_full,
        'hr10_clean': 100.0 * hr_clean,
        'unlearn_seconds': unlearn.compute_time,
        'trace_csv': trace_path,
    }

    # 공격 비율을 바꿔 가며 반복 호출되므로 GPU 메모리를 비워 둔다.
    del clean_generator, clean_graph, hr_ctx, model, unlearn
    torch.cuda.empty_cache()

    return summary

if __name__=='__main__':
    ##os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    summaries = []

    for attack in ATTACK_RATIOS:
        config_args = {}
        config_args['embed_size'] = 32
        config_args['batch_size'] = 2048
        config_args['epoch'] = 200
        config_args['data_path'] = 'Data/Process/'
        config_args['dataset'] = 'MovieLens'
        config_args['attack'] = attack
        config_args['k_hop'] = 1
        config_args['data_type'] = 'full'
        config_args['if_epoch'] = 200
        config_args['if_lr'] = 1e4
        config_args['if_init_std'] = 0
        config_args['seed'] = 1024
        config_args['lr'] = 1e-4
        config_args['regs'] = 0
        config_args['init_std'] = 1e-4
        config_args['n_fillers'] = int(os.environ.get('N_FILLERS', 150))
        config_args['if_epoch'] = int(os.environ.get('IF_EPOCH', 200))

        summaries.append(main(config_args))

    os.makedirs(RESULT_DIR, exist_ok=True)
    n_fillers = int(os.environ.get('N_FILLERS', 150))
    summary_path = os.path.join(RESULT_DIR, 'ifru_summary_filler%d.csv' % n_fillers)
    pd.DataFrame(summaries).to_csv(summary_path, index=False)

    print("\n" + "=" * 78)
    print("IFRU 요약 (필러 %d)" % n_fillers)
    print("=" * 78)
    print(pd.DataFrame(summaries)[[
        'attack_ratio', 'hr10_full', 'hr10_clean',
        'test_auc_full', 'test_auc_clean', 'unlearn_seconds']].to_string(index=False))
    print("\n저장:", summary_path)
    print("epoch별 기록: %s/ifru_trace_filler%d_attack*.csv" % (RESULT_DIR, n_fillers))
    print("\nAll IFRU runs finished.")
