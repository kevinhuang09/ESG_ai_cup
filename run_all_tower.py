import sys, os, random, time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from scipy.optimize import minimize
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
D = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME_1 = 'BAAI/bge-m3'
MODEL_NAME_2 = 'intfloat/multilingual-e5-large'
MODEL_NAME_3 = 'shibing624/text2vec-base-chinese'
SEED = int(os.environ.get('ESG_SEED', 42))
TAG = os.environ.get('ESG_TAG', 'cv10_soft_pseudo_labeling')
USE_OPENCC = os.environ.get('ESG_OPENCC', '0') == '1'
USE_TIMEFEAT = True
USE_CONCFEAT = True
NFOLD = 10
OUT = os.path.join(D, f'out_{TAG}')
SOFT_PSEUDO_THRESH = 0.85
PREV_BEST_OUT = os.path.join(D, 'out_cv10_tripletower_features_on')
PROMISE = ['Yes', 'No']
TIMELINE = ['already', 'within_2_years', 'between_2_and_5_years', 'more_than_5_years']
EVSTAT = ['Yes', 'No']
QUALITY = ['Clear', 'Not Clear', 'Misleading']
W = {'promise_status': 0.2, 'verification_timeline': 0.15, 'evidence_status': 0.3, 'evidence_quality': 0.35}
SCORER_A = {'promise_status': PROMISE, 'verification_timeline': TIMELINE + ['N/A'], 'evidence_status': EVSTAT + ['N/A'], 'evidence_quality': QUALITY + ['N/A']}
SCORER_B = {'promise_status': PROMISE, 'verification_timeline': TIMELINE, 'evidence_status': EVSTAT, 'evidence_quality': QUALITY}
HEADS = {'p': ('promise_status', PROMISE), 't': ('verification_timeline', TIMELINE), 'e': ('evidence_status', EVSTAT), 'q': ('evidence_quality', QUALITY)}

def set_seed(s):
    random.seed(s)
    np.random.seed(s)

def load_all():
    tr = pd.read_csv(os.path.join(D, 'vpesg_4k_train_1000.csv'), encoding='utf-8-sig')
    va = pd.read_csv(os.path.join(D, 'vpesg4k_val_1000.csv'), encoding='utf-8-sig')
    df = pd.concat([tr, va], ignore_index=True)
    df['sample_weight'] = 1.0
    for c in ['verification_timeline', 'evidence_status', 'evidence_quality']:
        df[c] = df[c].fillna('N/A').str.strip()
    df['promise_status'] = df['promise_status'].str.strip()
    te = pd.read_csv(os.path.join(D, 'vpesg4k_test_2000.csv'), encoding='utf-8-sig')
    if USE_TIMEFEAT or USE_CONCFEAT:
        print(f'\n[特徵工程啟動] 正在注入時間特徵({USE_TIMEFEAT})與概念特徵({USE_CONCFEAT})...', flush=True)
        from year_parse import inject_features
        fn = lambda r: inject_features(r['data'], r['pdf_url'], timefeat=USE_TIMEFEAT, concfeat=USE_CONCFEAT)
        df['data'] = df.apply(fn, axis=1)
        te = te.copy()
        te['data'] = te.apply(fn, axis=1)
        print('[特徵工程啟動] 注入完成！\n', flush=True)
    if USE_OPENCC:
        from opencc import OpenCC
        cc = OpenCC('t2s')
        df['data'] = df['data'].map(cc.convert)
        te = te.copy()
        te['data'] = te['data'].map(cc.convert)
    prob_path = os.path.join(PREV_BEST_OUT, 'test_probs.npz')
    if os.path.exists(prob_path):
        print(f'[偽標籤檢索] 找到歷史最佳預測：{prob_path}，啟用軟性權重篩選...', flush=True)
        probs = np.load(prob_path)
        conds = []
        pseudo_labels = {}
        mean_confidences = np.zeros(len(te))
        for k, (col, labs) in HEADS.items():
            p = probs[k]
            max_p = p.max(axis=1)
            pred_idx = p.argmax(axis=1)
            mean_confidences += max_p / len(HEADS)
            conds.append(max_p >= SOFT_PSEUDO_THRESH)
            pseudo_labels[col] = [labs[i] for i in pred_idx]
        final_cond = (np.sum(conds, axis=0) >= 3) & (mean_confidences >= SOFT_PSEUDO_THRESH)
        pseudo_idx = np.where(final_cond)[0]
        if len(pseudo_idx) > 0:
            print(f'🔥 [強力擴增] 成功以軟性門檻篩選出 {len(pseudo_idx)} 筆 Test 資料併入訓練集！', flush=True)
            te_pseudo = te.iloc[pseudo_idx].copy()
            te_pseudo['sample_weight'] = mean_confidences[pseudo_idx] * 0.8
            for col in pseudo_labels.keys():
                te_pseudo[col] = np.array(pseudo_labels[col])[pseudo_idx]
            df = pd.concat([df, te_pseudo], ignore_index=True)
            print(f'[數據更新] 新訓練集總數擴充至: {len(df)} 筆，包含加權偽標籤。\n', flush=True)
        else:
            print('⚠️ 軟性門檻下依然沒有樣本通過，保持純原始資料訓練。\n', flush=True)
    else:
        print(f'❌ 找不到歷史預測檔 {prob_path}，將以常規 10-Fold 運行。\n', flush=True)
    return (df, te)

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

def dual_score(df, preds):
    out = {}
    for nm, S in [('A', SCORER_A), ('B', SCORER_B)]:
        out[nm] = sum((W[f] * f1_score(df[f], preds[f], labels=l, average='macro', zero_division=0) for f, l in S.items()))
    out['mean'] = (out['A'] + out['B']) / 2
    return out

def optimize_blend(y_true, prob_a, prob_b):
    best_w = 0.5
    best_f1 = -1
    for w in np.linspace(0, 1, 101):
        p_blend = w * prob_a + (1 - w) * prob_b
        preds = np.argmax(p_blend, axis=1)
        score = f1_score(y_true, preds, average='macro', zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_w = w
    return best_w

def optimize_weights(y_true, probs):

    def loss_fn(weights):
        preds = np.argmax(probs * weights, axis=1)
        return -f1_score(y_true, preds, average='macro', zero_division=0)
    num_classes = probs.shape[1]
    initial_weights = np.ones(num_classes)
    res = minimize(loss_fn, initial_weights, method='Nelder-Mead', tol=0.001)
    return res.x

def train_best_lr(X, y, sw, seed):
    param_grid = {'C': [0.1, 0.5, 1.0, 5.0]}
    base_lr = LogisticRegression(class_weight='balanced', max_iter=1500, random_state=seed)
    search = GridSearchCV(base_lr, param_grid, scoring='f1_macro', cv=3, n_jobs=-1)
    search.fit(X, y, sample_weight=sw)
    return search.best_estimator_

def train_best_svm(X, y, sw, seed):
    param_grid = {'C': [0.05, 0.1, 0.5, 1.0]}
    base_svm = LinearSVC(class_weight='balanced', random_state=seed, dual=False, max_iter=2500)
    search = GridSearchCV(base_svm, param_grid, scoring='f1_macro', cv=3, n_jobs=-1)
    search.fit(X, y, sample_weight=sw)
    best_svc = search.best_estimator_
    calibrated = CalibratedClassifierCV(best_svc, cv=[(np.arange(len(y)), np.arange(len(y)))], ensemble=True)
    calibrated.fit(X, y, sample_weight=sw)
    return calibrated

def main():
    os.makedirs(OUT, exist_ok=True)
    set_seed(SEED)
    df, te = load_all()
    t_start = time.time()
    print(f'正在載入第一塔 {MODEL_NAME_1} ...', flush=True)
    embed_model_1 = SentenceTransformer(MODEL_NAME_1)
    X1_all = embed_model_1.encode(df['data'].tolist(), show_progress_bar=True, batch_size=32)
    X1_test = embed_model_1.encode(te['data'].tolist(), show_progress_bar=True, batch_size=32)
    del embed_model_1
    print(f'\n正在載入第二塔 {MODEL_NAME_2} ...', flush=True)
    embed_model_2 = SentenceTransformer(MODEL_NAME_2)
    df_data_me5 = ['passage: ' + str(text) for text in df['data'].tolist()]
    te_data_me5 = ['passage: ' + str(text) for text in te['data'].tolist()]
    X2_all = embed_model_2.encode(df_data_me5, show_progress_bar=True, batch_size=32)
    X2_test = embed_model_2.encode(te_data_me5, show_progress_bar=True, batch_size=32)
    del embed_model_2
    print(f'\n正在載入第三塔 {MODEL_NAME_3} ...', flush=True)
    embed_model_3 = SentenceTransformer(MODEL_NAME_3)
    X3_all = embed_model_3.encode(df['data'].tolist(), show_progress_bar=True, batch_size=32)
    X3_test = embed_model_3.encode(te['data'].tolist(), show_progress_bar=True, batch_size=32)
    del embed_model_3
    print(f'\n三塔向量全部提取完成！總耗時: {time.time() - t_start:.1f}s\n', flush=True)
    y_dict = {}
    for k, (col, labs) in HEADS.items():
        y_dict[k] = np.array([labs.index(v) if v in labs else -100 for v in df[col]])
    strat = df['promise_status'] + '|' + df['evidence_status'] + '|' + df['evidence_quality']
    vc = strat.value_counts()
    strat = strat.where(strat.map(vc) >= NFOLD, 'rare')
    skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
    oof_probs_t1_lr, oof_probs_t1_svm = ({k: np.zeros((len(df), len(labs))) for k, (_, labs) in HEADS.items()}, {k: np.zeros((len(df), len(labs))) for k, (_, labs) in HEADS.items()})
    oof_probs_t2_lr, oof_probs_t2_svm = ({k: np.zeros((len(df), len(labs))) for k, (_, labs) in HEADS.items()}, {k: np.zeros((len(df), len(labs))) for k, (_, labs) in HEADS.items()})
    oof_probs_t3_lr, oof_probs_t3_svm = ({k: np.zeros((len(df), len(labs))) for k, (_, labs) in HEADS.items()}, {k: np.zeros((len(df), len(labs))) for k, (_, labs) in HEADS.items()})
    te_prob_t1_lr, te_prob_t1_svm = ({k: np.zeros((len(te), len(labs))) for k, (_, labs) in HEADS.items()}, {k: np.zeros((len(te), len(labs))) for k, (_, labs) in HEADS.items()})
    te_prob_t2_lr, te_prob_t2_svm = ({k: np.zeros((len(te), len(labs))) for k, (_, labs) in HEADS.items()}, {k: np.zeros((len(te), len(labs))) for k, (_, labs) in HEADS.items()})
    te_prob_t3_lr, te_prob_t3_svm = ({k: np.zeros((len(te), len(labs))) for k, (_, labs) in HEADS.items()}, {k: np.zeros((len(te), len(labs))) for k, (_, labs) in HEADS.items()})
    sw_all = df['sample_weight'].to_numpy()
    print(f'--- 開始執行 {NFOLD}-Fold 加權 GridSearch 訓練 ---')
    for fold, (itr, iva) in enumerate(skf.split(df, strat)):
        t0 = time.time()
        for k, (col, labs) in HEADS.items():
            y_all = y_dict[k]
            y_tr, y_va = (y_all[itr], y_all[iva])
            train_mask = y_tr != -100
            y_tr_head = y_tr[train_mask]
            sw_tr_head = sw_all[itr][train_mask]
            X1_tr_head = X1_all[itr][train_mask]
            m_t1_lr = train_best_lr(X1_tr_head, y_tr_head, sw_tr_head, SEED + fold)
            m_t1_svm = train_best_svm(X1_tr_head, y_tr_head, sw_tr_head, SEED + fold)
            X2_tr_head = X2_all[itr][train_mask]
            m_t2_lr = train_best_lr(X2_tr_head, y_tr_head, sw_tr_head, SEED + fold)
            m_t2_svm = train_best_svm(X2_tr_head, y_tr_head, sw_tr_head, SEED + fold)
            X3_tr_head = X3_all[itr][train_mask]
            m_t3_lr = train_best_lr(X3_tr_head, y_tr_head, sw_tr_head, SEED + fold)
            m_t3_svm = train_best_svm(X3_tr_head, y_tr_head, sw_tr_head, SEED + fold)
            for idx, class_id in enumerate(m_t1_lr.classes_):
                oof_probs_t1_lr[k][iva, class_id] = m_t1_lr.predict_proba(X1_all[iva])[:, idx]
                oof_probs_t1_svm[k][iva, class_id] = m_t1_svm.predict_proba(X1_all[iva])[:, idx]
                te_prob_t1_lr[k][:, class_id] += m_t1_lr.predict_proba(X1_test)[:, idx] / NFOLD
                te_prob_t1_svm[k][:, class_id] += m_t1_svm.predict_proba(X1_test)[:, idx] / NFOLD
                oof_probs_t2_lr[k][iva, class_id] = m_t2_lr.predict_proba(X2_all[iva])[:, idx]
                oof_probs_t2_svm[k][iva, class_id] = m_t2_svm.predict_proba(X2_all[iva])[:, idx]
                te_prob_t2_lr[k][:, class_id] += m_t2_lr.predict_proba(X2_test)[:, idx] / NFOLD
                te_prob_t2_svm[k][:, class_id] += m_t2_svm.predict_proba(X2_test)[:, idx] / NFOLD
                oof_probs_t3_lr[k][iva, class_id] = m_t3_lr.predict_proba(X3_all[iva])[:, idx]
                oof_probs_t3_svm[k][iva, class_id] = m_t3_svm.predict_proba(X3_all[iva])[:, idx]
                te_prob_t3_lr[k][:, class_id] += m_t3_lr.predict_proba(X3_test)[:, idx] / NFOLD
                te_prob_t3_svm[k][:, class_id] += m_t3_svm.predict_proba(X3_test)[:, idx] / NFOLD
        print(f'Fold {fold + 1}/{NFOLD} 訓練完畢 | 耗時: {time.time() - t0:.0f}s', flush=True)
    print('\n--- 開始執行階層式模型融合與閾值最佳化 ---')
    oof_probs_final = {}
    te_prob_final = {}
    for k, (col, _) in HEADS.items():
        valid_mask = y_dict[k] != -100
        y_v = y_dict[k][valid_mask]
        w1 = optimize_blend(y_v, oof_probs_t1_lr[k][valid_mask], oof_probs_t1_svm[k][valid_mask])
        t1_oof = w1 * oof_probs_t1_lr[k] + (1 - w1) * oof_probs_t1_svm[k]
        t1_te = w1 * te_prob_t1_lr[k] + (1 - w1) * te_prob_t1_svm[k]
        w2 = optimize_blend(y_v, oof_probs_t2_lr[k][valid_mask], oof_probs_t2_svm[k][valid_mask])
        t2_oof = w2 * oof_probs_t2_lr[k] + (1 - w2) * oof_probs_t2_svm[k]
        t2_te = w2 * te_prob_t2_lr[k] + (1 - w2) * te_prob_t2_svm[k]
        w3 = optimize_blend(y_v, oof_probs_t3_lr[k][valid_mask], oof_probs_t3_svm[k][valid_mask])
        t3_oof = w3 * oof_probs_t3_lr[k] + (1 - w3) * oof_probs_t3_svm[k]
        t3_te = w3 * te_prob_t3_lr[k] + (1 - w3) * te_prob_t3_svm[k]
        w_12 = optimize_blend(y_v, t1_oof[valid_mask], t2_oof[valid_mask])
        t12_oof = w_12 * t1_oof + (1.0 - w_12) * t2_oof
        t12_te = w_12 * t1_te + (1.0 - w_12) * t2_te
        w_final = optimize_blend(y_v, t12_oof[valid_mask], t3_oof[valid_mask])
        print(f'[{col}] 融合狀況 -> 雙塔基底(T12)佔 {w_final:.2f} | 中文神塔(T3)佔 {1.0 - w_final:.2f}')
        final_oof = w_final * t12_oof + (1.0 - w_final) * t3_oof
        final_te = w_final * t12_te + (1.0 - w_final) * t3_te
        if final_oof.shape[1] > 1:
            w_thresh = optimize_weights(y_v, final_oof[valid_mask])
            w_thresh = w_thresh / np.max(np.abs(w_thresh))
            final_oof = final_oof * w_thresh
            final_te = final_te * w_thresh
        oof_probs_final[k] = final_oof
        te_prob_final[k] = final_te
    df_clean = df.iloc[:2000]
    oof_clean = {k: v[:2000] for k, v in oof_probs_final.items()}
    s_final = dual_score(df_clean, decode(oof_clean))
    print(f"\n🏆 [Soft Pseudo-Labeling 10Fold] OOF ({TAG}): A {s_final['A']:.4f} B {s_final['B']:.4f} mean {s_final['mean']:.4f}")
    np.savez(os.path.join(OUT, 'oof_logits.npz'), **oof_clean)
    np.savez(os.path.join(OUT, 'test_probs.npz'), **te_prob_final)
    sub = pd.DataFrame({'id': te['id'], **decode(te_prob_final)})
    sub = sub[['id', 'promise_status', 'verification_timeline', 'evidence_status', 'evidence_quality']]
    assert len(sub) == 2000 and list(sub['id']) == list(range(12001, 14001))
    sub.to_csv(os.path.join(OUT, f'submission_{TAG}.csv'), index=False, encoding='utf-8')
    print(f'\n✅ 成功！軟性加權偽標籤版 10-Fold Submission 已輸出至: out_{TAG}/submission_{TAG}.csv')
if __name__ == '__main__':
    main()