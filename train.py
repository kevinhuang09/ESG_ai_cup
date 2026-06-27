import os
import sys
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import softmax
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
BACKBONE    = os.environ.get('ESG_MODEL', 'hfl/chinese-roberta-wwm-ext')
SEQ_LEN     = int(os.environ.get('ESG_MAXLEN', 384))
INFER_BS    = 32 if SEQ_LEN > 384 else 64
TRAIN_BS    = int(os.environ.get('ESG_BATCH', 16))
NUM_EPOCHS  = int(os.environ.get('ESG_EPOCHS', 5))
BASE_LR     = float(os.environ.get('ESG_LR', 2e-05))
RNG_SEED    = int(os.environ.get('ESG_SEED', 42))
RUN_TAG     = os.environ.get('ESG_TAG', 'cv_rbt')
TRAD2SIMP   = os.environ.get('ESG_OPENCC', '0') == '1'
CLS_WEIGHT  = os.environ.get('ESG_CLASSWEIGHT', '0') == '1'
CW_CEILING  = float(os.environ.get('ESG_CW_CAP', 4.0))
GRAD_ACCUM  = os.environ.get('ESG_GRADCKPT', '0') == '1'
ACTIVE_HEADS = os.environ.get('ESG_LOSS_HEADS', 'p,t,e,q').split(',')
ADD_YEAR    = os.environ.get('ESG_TIMEFEAT', '0') == '1'
ADD_QUANT   = os.environ.get('ESG_CONCFEAT', '0') == '1'
USE_SPAN    = os.environ.get('ESG_SPANAUX', '0') == '1'
SPAN_ALPHA  = float(os.environ.get('ESG_SPAN_W', 0.5))
NUM_FOLDS   = 5
SAVE_DIR    = os.path.join(BASE_DIR, f'out_{RUN_TAG}')

LABEL_P = ['Yes', 'No']
LABEL_T = ['already', 'within_2_years', 'between_2_and_5_years', 'more_than_5_years']
LABEL_E = ['Yes', 'No']
LABEL_Q = ['Clear', 'Not Clear', 'Misleading']

FIELD_WEIGHT = {
    'promise_status': 0.2,
    'verification_timeline': 0.15,
    'evidence_status': 0.3,
    'evidence_quality': 0.35,
}

EVAL_SCHEMA_A = {
    'promise_status':       LABEL_P,
    'verification_timeline': LABEL_T + ['N/A'],
    'evidence_status':      LABEL_E + ['N/A'],
    'evidence_quality':     LABEL_Q + ['N/A'],
}
EVAL_SCHEMA_B = {
    'promise_status':       LABEL_P,
    'verification_timeline': LABEL_T,
    'evidence_status':      LABEL_E,
    'evidence_quality':     LABEL_Q,
}

TASK_MAP = {
    'p': ('promise_status',        LABEL_P),
    't': ('verification_timeline', LABEL_T),
    'e': ('evidence_status',       LABEL_E),
    'q': ('evidence_quality',      LABEL_Q),
}


def fix_rng(n):
    random.seed(n)
    np.random.seed(n)
    torch.manual_seed(n)
    torch.cuda.manual_seed_all(n)


def apply_label_hierarchy(raw):
    promise  = np.array(LABEL_P)[raw['p'].argmax(1)]
    timeline = np.array(LABEL_T)[raw['t'].argmax(1)].astype(object)
    evidence = np.array(LABEL_E)[raw['e'].argmax(1)].astype(object)
    quality  = np.array(LABEL_Q)[raw['q'].argmax(1)].astype(object)

    no_promise = promise == 'No'
    timeline[no_promise] = 'N/A'
    evidence[no_promise] = 'N/A'
    quality[no_promise | (evidence == 'No')] = 'N/A'

    return {
        'promise_status':        promise,
        'verification_timeline': timeline,
        'evidence_status':       evidence,
        'evidence_quality':      quality,
    }


def compute_dual_f1(reference_df, predictions):
    scores = {}
    for schema_name, schema in [('A', EVAL_SCHEMA_A), ('B', EVAL_SCHEMA_B)]:
        weighted = sum(
            FIELD_WEIGHT[field] * f1_score(
                reference_df[field], predictions[field],
                labels=allowed, average='macro', zero_division=0
            )
            for field, allowed in schema.items()
        )
        scores[schema_name] = weighted
    scores['mean'] = (scores['A'] + scores['B']) / 2
    return scores


@torch.no_grad()
def run_inference(net, loader, device):
    net.eval()
    collected = {key: [] for key in TASK_MAP}
    for batch in loader:
        ids  = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _ = net(ids, mask)
        for key in TASK_MAP:
            collected[key].append(logits[key].float().cpu())
    return {key: torch.cat(tensors).numpy() for key, tensors in collected.items()}


def read_datasets():
    part_train = pd.read_csv(os.path.join(BASE_DIR, 'vpesg_4k_train_1000.csv'), encoding='utf-8-sig')
    part_val   = pd.read_csv(os.path.join(BASE_DIR, 'vpesg4k_val_1000.csv'),   encoding='utf-8-sig')
    combined   = pd.concat([part_train, part_val], ignore_index=True)

    nullable_cols = ['verification_timeline', 'evidence_status', 'evidence_quality']
    for col in nullable_cols:
        combined[col] = combined[col].fillna('N/A').str.strip()
    combined['promise_status'] = combined['promise_status'].str.strip()

    holdout = pd.read_csv(os.path.join(BASE_DIR, 'vpesg4k_test_2000.csv'), encoding='utf-8-sig')

    if ADD_YEAR or ADD_QUANT:
        from year_parse import inject_features
        augment = lambda row: inject_features(row['data'], row['pdf_url'], timefeat=ADD_YEAR, concfeat=ADD_QUANT)
        combined['data'] = combined.apply(augment, axis=1)
        holdout = holdout.copy()
        holdout['data'] = holdout.apply(augment, axis=1)
        print(f"[feat] year={ADD_YEAR} quant={ADD_QUANT}; sample: {combined['data'].iloc[0][:36]}", flush=True)

    if TRAD2SIMP:
        from opencc import OpenCC
        converter = OpenCC('t2s')
        combined['data'] = combined['data'].map(converter.convert)
        holdout = holdout.copy()
        holdout['data'] = holdout['data'].map(converter.convert)

    return combined, holdout


def build_span_targets(text_list, evidence_list, offset_list):
    result = torch.full((len(text_list), SEQ_LEN), -100, dtype=torch.long)
    for idx, (text, ev_raw, token_offsets) in enumerate(zip(text_list, evidence_list, offset_list)):
        ev = str(ev_raw).strip()
        if ev in ('', 'nan', 'N/A'):
            continue
        span_ranges = []
        for segment in (seg.strip() for seg in ev.split('｜')):
            if not segment:
                continue
            start = text.find(segment)
            if start >= 0:
                span_ranges.append((start, start + len(segment)))
        for tok_idx, (char_start, char_end) in enumerate(token_offsets):
            if char_end <= char_start:
                continue
            inside = any(char_start < r_end and char_end > r_start for r_start, r_end in span_ranges)
            result[idx, tok_idx] = 1 if inside else 0
    return result


class ESGDataset(Dataset):
    def __init__(self, frame, tokenizer, is_labeled):
        self.is_labeled = is_labeled
        need_offsets = is_labeled and USE_SPAN
        enc = tokenizer(
            list(frame['data']),
            truncation=True,
            max_length=SEQ_LEN,
            padding='max_length',
            return_tensors='pt',
            return_offsets_mapping=need_offsets,
        )
        self.token_ids   = enc['input_ids']
        self.attn_mask   = enc['attention_mask']
        if is_labeled:
            self.labels = {}
            for key, (col, label_set) in TASK_MAP.items():
                self.labels[key] = torch.tensor(
                    [label_set.index(v) if v in label_set else -100 for v in frame[col]]
                )
            if USE_SPAN:
                self.span_targets = build_span_targets(
                    list(frame['data']),
                    list(frame['evidence_string']),
                    enc['offset_mapping'].tolist(),
                )

    def __len__(self):
        return self.token_ids.shape[0]

    def __getitem__(self, idx):
        sample = {
            'input_ids':      self.token_ids[idx],
            'attention_mask': self.attn_mask[idx],
        }
        if self.is_labeled:
            for key in TASK_MAP:
                sample[f'y_{key}'] = self.labels[key][idx]
            if USE_SPAN:
                sample['span'] = self.span_targets[idx]
        return sample


class MultiHeadEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(BACKBONE)
        if GRAD_ACCUM:
            self.transformer.gradient_checkpointing_enable()
            self.transformer.config.use_cache = False
        hidden = self.transformer.config.hidden_size
        self.dropout    = nn.Dropout(0.1)
        self.classifiers = nn.ModuleDict({
            key: nn.Linear(hidden, len(label_set))
            for key, (_, label_set) in TASK_MAP.items()
        })
        self.span_classifier = nn.Linear(hidden, 2) if USE_SPAN else None

    def forward(self, input_ids, attention_mask):
        hidden_states = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        cls_repr = self.dropout(hidden_states[:, 0])
        task_logits = {key: clf(cls_repr) for key, clf in self.classifiers.items()}
        span_logits = self.span_classifier(hidden_states) if self.span_classifier is not None else None
        return task_logits, span_logits


def run_fold(fold_idx, tr_frame, va_frame, tokenizer, loader_te, device, fold_logits, test_acc):
    fold_start = time.time()
    fix_rng(RNG_SEED + fold_idx)

    loader_tr = DataLoader(ESGDataset(tr_frame, tokenizer, True),  batch_size=TRAIN_BS, shuffle=True)
    loader_va = DataLoader(ESGDataset(va_frame, tokenizer, True),  batch_size=INFER_BS)

    net = MultiHeadEncoder().to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=BASE_LR, weight_decay=0.01)
    total_steps = len(loader_tr) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    loss_fn = {}
    for key, (col, label_set) in TASK_MAP.items():
        class_w = None
        if CLS_WEIGHT:
            counts   = tr_frame[col].value_counts()
            n_valid  = sum(counts.get(lbl, 0) for lbl in label_set)
            class_w  = torch.tensor(
                [min(n_valid / (len(label_set) * max(counts.get(lbl, 0), 1)), CW_CEILING) for lbl in label_set],
                dtype=torch.float32, device=device,
            )
        loss_fn[key] = nn.CrossEntropyLoss(ignore_index=-100, weight=class_w)

    if CLS_WEIGHT and fold_idx == 0:
        for key, (_, label_set) in TASK_MAP.items():
            weights_str = ', '.join(f'{l}={w:.2f}' for l, w in zip(label_set, loss_fn[key].weight.tolist()))
            print(f'  cw[{key}]: {weights_str}')

    span_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(NUM_EPOCHS):
        net.train()
        for step, batch in enumerate(loader_tr):
            ids  = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                task_out, span_out = net(ids, mask)
                total_loss = sum(
                    loss_fn[key](task_out[key], batch[f'y_{key}'].to(device))
                    for key in TASK_MAP
                )
                if USE_SPAN:
                    total_loss = total_loss + SPAN_ALPHA * span_loss_fn(
                        span_out.reshape(-1, 2),
                        batch['span'].reshape(-1).to(device),
                    )

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            if (step + 1) % 100 == 0:
                elapsed = time.time() - fold_start
                print(f'  f{fold_idx} ep{epoch+1} step {step+1}/{len(loader_tr)} loss {total_loss.item():.3f} ({elapsed:.0f}s)', flush=True)

    va_out = run_inference(net, loader_va, device)
    for key in TASK_MAP:
        fold_logits[key][va_frame.index] = va_out[key]

    te_out = run_inference(net, loader_te, device)
    for key in TASK_MAP:
        test_acc[key] += softmax(te_out[key], axis=1) / NUM_FOLDS

    fold_score = compute_dual_f1(va_frame, apply_label_hierarchy({k: va_out[k] for k in TASK_MAP}))
    elapsed = time.time() - fold_start
    print(f"fold{fold_idx}: A {fold_score['A']:.4f} B {fold_score['B']:.4f} mean {fold_score['mean']:.4f} | {elapsed:.0f}s", flush=True)

    del net
    torch.cuda.empty_cache()


def validate_submission(sub):
    assert len(sub) == 2000 and list(sub['id']) == list(range(12001, 14001))
    assert sub.notna().all().all()
    no_promise = sub['promise_status'] == 'No'
    assert (sub.loc[no_promise, ['verification_timeline', 'evidence_status', 'evidence_quality']] == 'N/A').all().all()
    assert (sub.loc[sub['evidence_status'] == 'No', 'evidence_quality'] == 'N/A').all()


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_df, test_df = read_datasets()
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    loader_te = DataLoader(ESGDataset(test_df, tokenizer, False), batch_size=INFER_BS)

    fold_key = all_df['promise_status'] + '|' + all_df['evidence_status'] + '|' + all_df['evidence_quality']
    key_counts = fold_key.value_counts()
    fold_key = fold_key.where(fold_key.map(key_counts) >= NUM_FOLDS, 'rare')

    splitter    = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RNG_SEED)
    fold_logits = {key: np.zeros((len(all_df), len(lset))) for key, (_, lset) in TASK_MAP.items()}
    test_acc    = {key: np.zeros((len(test_df), len(lset))) for key, (_, lset) in TASK_MAP.items()}

    for fold_idx, (tr_idx, va_idx) in enumerate(splitter.split(all_df, fold_key)):
        run_fold(
            fold_idx,
            all_df.iloc[tr_idx],
            all_df.iloc[va_idx],
            tokenizer,
            loader_te,
            device,
            fold_logits,
            test_acc,
        )

    oof_score = compute_dual_f1(all_df, apply_label_hierarchy(fold_logits))
    print(f"\nOOF ({RUN_TAG}, {BACKBONE}, ep{NUM_EPOCHS}): A {oof_score['A']:.4f} B {oof_score['B']:.4f} mean {oof_score['mean']:.4f}")

    np.savez(os.path.join(SAVE_DIR, 'oof_logits.npz'),  **fold_logits)
    np.savez(os.path.join(SAVE_DIR, 'test_probs.npz'),  **test_acc)

    submission = pd.DataFrame({'id': test_df['id'], **apply_label_hierarchy(test_acc)})
    submission = submission[['id', 'promise_status', 'verification_timeline', 'evidence_status', 'evidence_quality']]
    validate_submission(submission)

    out_path = os.path.join(SAVE_DIR, f'submission_{RUN_TAG}.csv')
    submission.to_csv(out_path, index=False, encoding='utf-8')
    print(f'submission written: out_{RUN_TAG}/submission_{RUN_TAG}.csv')


if __name__ == '__main__':
    main()
