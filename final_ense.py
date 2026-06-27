import sys, os, random, time
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from scipy.special import softmax
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
D = os.path.dirname(os.path.abspath(__file__))
ROBERTA_SEEDS = ['cv_rbtL_tf', 'cv_rbtL_tf_s1', 'cv_rbtL_tf_s2', 'cv_rbtL_tf_s3']
BGE_CONFIGS = ['bgeL_ep8_lr1e5', 'bgeL_ep10_lr1e5', 'bgeL_ep8_lr2e5', 'bgeL_ep10_lr2e5']
ELS_DIR = os.path.join(D, 'out_cv_els_tf')
TOWER_DIR = os.path.join(D, 'out_cv10_soft_pseudo_labeling')
MAIN_CONF_THRES = 0.6
EXP_CONF_THRES = 0.8
W_ROBERTA = 0.45
W_BGE = 0.55
PROMISE = ['Yes', 'No']
TIMELINE = ['already', 'within_2_years', 'between_2_and_5_years', 'more_than_5_years']
EVSTAT = ['Yes', 'No']
QUALITY = ['Clear', 'Not Clear', 'Misleading']
W_SCORE = {'promise_status': 0.2, 'verification_timeline': 0.15, 'evidence_status': 0.3, 'evidence_quality': 0.35}
SCORER_A = {'promise_status': PROMISE, 'verification_timeline': TIMELINE + ['N/A'], 'evidence_status': EVSTAT + ['N/A'], 'evidence_quality': QUALITY + ['N/A']}
SCORER_B = {'promise_status': PROMISE, 'verification_timeline': TIMELINE, 'evidence_status': EVSTAT, 'evidence_quality': QUALITY}

def dual_score(df, preds):
    out = {}
    for nm, S in [('A', SCORER_A), ('B', SCORER_B)]:
        out[nm] = sum((W_SCORE[f] * f1_score(df[f], preds[f], labels=l, average='macro', zero_division=0) for f, l in S.items()))
    out['mean'] = (out['A'] + out['B']) / 2
    return out

def decode(P):
    p = np.array(PROMISE)[P['p'].argmax(1)]
    t = np.array(TIMELINE)[P['t'].argmax(1)].astype(object)
    e = np.array(EVSTAT)[P['e'].argmax(1)].astype(object)
    q = np.array(QUALITY)[P['q'].argmax(1)].astype(object)
    no = p == 'No'
    t[no] = 'N/A'
    e[no] = 'N/A'
    q[no | (e == 'No')] = 'N/A'
    return {'promise_status': p, 'verification_timeline': t, 'evidence_status': e, 'evidence_quality': q}

def to_probs(arr):
    if np.min(arr) < 0 or not np.isclose(np.sum(arr[0]), 1.0, atol=0.01):
        e_x = np.exp(arr - np.max(arr, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
    return arr
print('讀取原始乾淨標籤中...')
tr = pd.read_csv(os.path.join(D, 'vpesg_4k_train_1000.csv'), encoding='utf-8-sig')
va = pd.read_csv(os.path.join(D, 'vpesg4k_val_1000.csv'), encoding='utf-8-sig')
df_clean = pd.concat([tr, va], ignore_index=True).iloc[:2000]
for c in ['verification_timeline', 'evidence_status', 'evidence_quality']:
    df_clean[c] = df_clean[c].fillna('N/A').str.strip()
df_clean['promise_status'] = df_clean['promise_status'].str.strip()
print(f'\n🔄 正在計算 OOF 雙巨頭融合分數 (RoBERTa*{W_ROBERTA} + BGE*{W_BGE} -> 專家末端救場)...')
oof_prob_final = {k: 0.0 for k in ('p', 't', 'e', 'q')}
rbt_oof = {k: 0.0 for k in ('p', 't', 'e', 'q')}
for tag in ROBERTA_SEEDS:
    z = np.load(os.path.join(D, f'out_{tag}', 'oof_logits.npz'))
    for k in rbt_oof:
        rbt_oof[k] += to_probs(z[k][:2000]) / len(ROBERTA_SEEDS)
bge_oof = {k: 0.0 for k in ('p', 't', 'e', 'q')}
for tag in BGE_CONFIGS:
    z = np.load(os.path.join(D, f'out_{tag}', 'oof_logits.npz'))
    for k in bge_oof:
        bge_oof[k] += to_probs(z[k][:2000]) / len(BGE_CONFIGS)
main_oof = {k: rbt_oof[k] * W_ROBERTA + bge_oof[k] * W_BGE for k in ('p', 't', 'e', 'q')}
els_oof = {k: to_probs(np.load(os.path.join(ELS_DIR, 'oof_logits.npz'))[k][:2000]) for k in ('p', 't', 'e', 'q')}
twr_oof = {k: to_probs(np.load(os.path.join(TOWER_DIR, 'oof_logits.npz'))[k][:2000]) for k in ('p', 't', 'e', 'q')}
oof_replace_counts_els = {}
oof_replace_counts_twr = {}
for k in oof_prob_final:
    final_p = np.copy(main_oof[k])
    main_conf = np.max(main_oof[k], axis=1)
    els_conf = np.max(els_oof[k], axis=1)
    twr_conf = np.max(twr_oof[k], axis=1)
    els_cnt = 0
    twr_cnt = 0
    for i in range(len(final_p)):
        if main_conf[i] < MAIN_CONF_THRES:
            els_ok = els_conf[i] > EXP_CONF_THRES
            twr_ok = twr_conf[i] > EXP_CONF_THRES
            if els_ok and twr_ok:
                if els_conf[i] >= twr_conf[i]:
                    final_p[i] = els_oof[k][i]
                    els_cnt += 1
                else:
                    final_p[i] = twr_oof[k][i]
                    twr_cnt += 1
            elif els_ok:
                final_p[i] = els_oof[k][i]
                els_cnt += 1
            elif twr_ok:
                final_p[i] = twr_oof[k][i]
                twr_cnt += 1
    oof_prob_final[k] = final_p
    oof_replace_counts_els[k] = els_cnt
    oof_replace_counts_twr[k] = twr_cnt
print(f"📊 [OOF 救場統計 - els]: p: {oof_replace_counts_els['p']}筆 | t: {oof_replace_counts_els['t']}筆 | e: {oof_replace_counts_els['e']}筆 | q: {oof_replace_counts_els['q']}筆")
print(f"📊 [OOF 救場統計 - Tower]: p: {oof_replace_counts_twr['p']}筆 | t: {oof_replace_counts_twr['t']}筆 | e: {oof_replace_counts_twr['e']}筆 | q: {oof_replace_counts_twr['q']}筆")
oof_preds = decode(oof_prob_final)
s_final = dual_score(df_clean, oof_preds)
print(f"🏆 [雙巨頭+專家驗證成功] OOF 總分數 -> A: {s_final['A']:.4f} | B: {s_final['B']:.4f} | 🔥 mean: {s_final['mean']:.4f}\n")
print(f'🔮 正在進行測試集雙巨頭動態加權集成 (RoBERTa*{W_ROBERTA} + BGE*{W_BGE})...')
te_prob_final = {k: 0.0 for k in ('p', 't', 'e', 'q')}
rbt_tep = {k: 0.0 for k in ('p', 't', 'e', 'q')}
for tag in ROBERTA_SEEDS:
    z = np.load(os.path.join(D, f'out_{tag}', 'test_probs.npz'))
    for k in rbt_tep:
        rbt_tep[k] += to_probs(z[k]) / len(ROBERTA_SEEDS)
bge_tep = {k: 0.0 for k in ('p', 't', 'e', 'q')}
for tag in BGE_CONFIGS:
    z = np.load(os.path.join(D, f'out_{tag}', 'test_probs.npz'))
    for k in bge_tep:
        bge_tep[k] += to_probs(z[k]) / len(BGE_CONFIGS)
main_tep = {k: rbt_tep[k] * W_ROBERTA + bge_tep[k] * W_BGE for k in ('p', 't', 'e', 'q')}
els_tep = {k: to_probs(np.load(os.path.join(ELS_DIR, 'test_probs.npz'))[k]) for k in ('p', 't', 'e', 'q')}
twr_tep = {k: to_probs(np.load(os.path.join(TOWER_DIR, 'test_probs.npz'))[k]) for k in ('p', 't', 'e', 'q')}
test_replace_els = {}
test_replace_twr = {}
for k in te_prob_final:
    final_p = np.copy(main_tep[k])
    main_conf = np.max(main_tep[k], axis=1)
    els_conf = np.max(els_tep[k], axis=1)
    twr_conf = np.max(twr_tep[k], axis=1)
    els_cnt = 0
    twr_cnt = 0
    for i in range(len(final_p)):
        if main_conf[i] < MAIN_CONF_THRES:
            els_ok = els_conf[i] > EXP_CONF_THRES
            twr_ok = twr_conf[i] > EXP_CONF_THRES
            if els_ok and twr_ok:
                if els_conf[i] >= twr_conf[i]:
                    final_p[i] = els_tep[k][i]
                    els_cnt += 1
                else:
                    final_p[i] = twr_tep[k][i]
                    twr_cnt += 1
            elif els_ok:
                final_p[i] = els_tep[k][i]
                els_cnt += 1
            elif twr_ok:
                final_p[i] = twr_tep[k][i]
                twr_cnt += 1
    te_prob_final[k] = final_p
    test_replace_els[k] = els_cnt
    test_replace_twr[k] = twr_cnt
print(f"📊 [Test 救場統計 - els]: p: {test_replace_els['p']}筆 | t: {test_replace_els['t']}筆 | e: {test_replace_els['e']}筆 | q: {test_replace_els['q']}筆")
print(f"📊 [Test 救場統計 - Tower]: p: {test_replace_twr['p']}筆 | t: {test_replace_twr['t']}筆 | e: {test_replace_twr['e']}筆 | q: {test_replace_twr['q']}筆")
te = pd.read_csv(os.path.join(D, 'vpesg4k_test_2000.csv'), encoding='utf-8-sig')
sub = pd.DataFrame({'id': te['id'], **decode(te_prob_final)})
sub = sub[['id', 'promise_status', 'verification_timeline', 'evidence_status', 'evidence_quality']]
assert len(sub) == 2000 and list(sub['id']) == list(range(12001, 14001))
assert sub.notna().all().all()
no = sub['promise_status'] == 'No'
assert (sub.loc[no, ['verification_timeline', 'evidence_status', 'evidence_quality']] == 'N/A').all().all()
assert (sub.loc[sub['evidence_status'] == 'No', 'evidence_quality'] == 'N/A').all()
out_path = os.path.join(D, 'submission_TITANS_dynamic_confidence.csv')
sub.to_csv(out_path, index=False, encoding='utf-8')
print(f'\n🎉 【世紀帝國合體版】雙巨頭主力（RoBERTa+BGE加權）+ 雙專家救場提交檔已完美生成：\n➡️ {out_path} (格式與強規則驗證完全通過！)')