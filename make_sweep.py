import os
import sys

SRC = 'ifru_measure_v2.py'
DST = 'ifru_sweep.py'

HEAD = "        self.p = Variable(torch.empty([unlearn_grad.shape[0],1])).cuda()\n"
TAIL = '            print("[trace] epoch별 기록 저장:", trace_path)\n'

NEW = '''        # ================= 여러 (if_lr, lambda) 조합을 한 번의 setup으로 =================
        _lrs = os.environ.get('IF_LRS')
        _lams = os.environ.get('LAMBDAS', '0')
        lr_list = [float(x) for x in _lrs.split(',') if x.strip()] if _lrs else [self.if_lr]
        lam_list = [float(x) for x in _lams.split(',') if x.strip()]
        configs = [(a, b) for a in lr_list for b in lam_list]
        eval_every = int(os.environ.get('EVAL_EVERY', 1))

        orig_u_vals = model.embedding_user.weight.data[nei_users].clone()
        orig_i_vals = model.embedding_item.weight.data[nei_items].clone()

        _sync()
        self.compute_time += time.time() - _t_setup
        setup_time = self.compute_time
        print("[time] 루프 진입 전 준비(기울기·헤시안 재료 계산): %.3f초" % setup_time)
        print("[sweep] 조합 %d개, 평가 주기 %d epoch: %s" % (len(configs), eval_every, configs))

        def grad_goal_damped(grad, vec, lam):
            # 감쇠항을 넣으면 (H + lam*I) p = g 를 푸는 셈이 된다.
            res = hvp(grad, vec).unsqueeze(-1) - unlearn_grad.detach()
            if lam:
                res = res + lam * vec.detach()
            return res

        for cfg_lr, cfg_lam in configs:
            # 직전 조합이 건드린 임베딩을 원래 값으로 되돌린다
            model.embedding_user.weight.data[nei_users] = orig_u_vals.clone()
            model.embedding_item.weight.data[nei_items] = orig_i_vals.clone()

            self.p = Variable(torch.empty([unlearn_grad.shape[0], 1])).cuda()
            nn.init.uniform_(self.p, -self.range, self.range)
            opt = torch.optim.Adam([self.p], lr=cfg_lr)

            tag = 'lr{:g}_lam{:g}'.format(cfg_lr, cfg_lam)
            loop_time = 0.0
            trace_rows = []
            print("")
            print("=" * 72)
            print("[sweep] if_lr=%g  lambda=%g" % (cfg_lr, cfg_lam))
            print("=" * 72)

            for if_ep in range(self.if_epoch):
                # ---- 보정 연산. 시간에 포함 ----
                _sync()
                s_time = time.time()
                opt.zero_grad()
                self.p.grad = grad_goal_damped(total_grad, self.p, cfg_lam)
                opt.step()
                with torch.no_grad():
                    un_ui_para_temp = un_ui_para + 1.0 / data_generator.n_train * self.p.squeeze()
                    model.embedding_user.weight.data[nei_users] = un_ui_para_temp[:u_para_num].reshape(-1, u_para.shape[-1]).data + 0
                    model.embedding_item.weight.data[nei_items] = un_ui_para_temp[u_para_num:].reshape(-1, i_para.shape[-1]).data + 0
                _sync()
                step_time = time.time() - s_time
                loop_time += step_time
                # ---- 보정 끝. 아래는 계측이라 시간에 안 넣는다 ----

                is_last = (if_ep == self.if_epoch - 1)
                if (if_ep % eval_every != 0) and not is_last:
                    continue

                with torch.no_grad():
                    p_norm = float(torch.norm(self.p).item())
                    residual = float(torch.norm(self.p.grad).item())

                    valid_auc, _, _, test_auc, _, _ = get_eval_result(data_generator, model, mask)

                    hr_full = float('nan')
                    hr_clean = float('nan')
                    v_clean = float('nan')
                    t_clean = float('nan')
                    if hr_ctx is not None:
                        hr_full = compute_hr10(model, model.Graph, hr_ctx)
                        if clean_graph is not None:
                            hr_clean = compute_hr10(model, clean_graph, hr_ctx)
                    if clean_graph is not None:
                        _og = model.Graph
                        model.Graph = clean_graph
                        v_clean, _, _, t_clean, _, _ = get_eval_result(data_generator, model, mask)
                        model.Graph = _og

                    print("ep %3d | clean test auc %.6f  HR@10 %.4f%% | |p| %.4e  resid %.4e"
                          % (if_ep, t_clean, 100.0 * hr_clean, p_norm, residual))

                    trace_rows.append({
                        'if_lr': cfg_lr,
                        'lam': cfg_lam,
                        'epoch': if_ep,
                        'step_time': step_time,
                        'loop_time': loop_time,
                        'setup_time': setup_time,
                        'valid_auc_full': valid_auc,
                        'test_auc_full': test_auc,
                        'valid_auc_clean': v_clean,
                        'test_auc_clean': t_clean,
                        'hr10_full': hr_full,
                        'hr10_clean': hr_clean,
                        'p_norm': p_norm,
                        'residual': residual,
                    })

            torch.save(model.state_dict(), self.save_name)
            print("[time] %s — 보정 루프 %.3f초 (준비 %.3f초는 조합끼리 공유)" % (tag, loop_time, setup_time))

            if trace_path is not None and len(trace_rows):
                base, ext = os.path.splitext(trace_path)
                out = '{}_{}{}'.format(base, tag, ext)
                d = os.path.dirname(out)
                if d:
                    os.makedirs(d, exist_ok=True)
                pd.DataFrame(trace_rows).to_csv(out, index=False)
                print("[trace] 저장:", out)
'''

if not os.path.exists(SRC):
    sys.exit('!! %s 가 없습니다. 저장소 루트에서 실행하세요.' % SRC)
s = open(SRC, encoding='utf-8').read()
if HEAD not in s:
    sys.exit('!! 치환 시작 지점을 못 찾았습니다.')
if TAIL not in s:
    sys.exit('!! 치환 끝 지점을 못 찾았습니다.')
i = s.index(HEAD)
j = s.index(TAIL) + len(TAIL)
open(DST, 'w', encoding='utf-8').write(s[:i] + NEW + s[j:])
print('>> %s 생성 (%d줄 -> %d줄)' % (DST, s.count('\n'), (s[:i] + NEW + s[j:]).count('\n')))
