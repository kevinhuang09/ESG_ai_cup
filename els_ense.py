import os

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import f1_score

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
RUN_TAGS  = ['cv_els_tf', 'cv_els_tf_s1', 'cv_els_tf_s2', 'cv_els_tf_s3']
NUM_RUNS  = len(RUN_TAGS)
print(f'seeds in ensemble: {NUM_RUNS} -> {RUN_TAGS}')

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
    'promise_status':        LABEL_P,
    'verification_timeline': LABEL_T + ['N/A'],
    'evidence_status':       LABEL_E + ['N/A'],
    'evidence_quality':      LABEL_Q + ['N/A'],
}
EVAL_SCHEMA_B = {
    'promise_status':        LABEL_P,
    'verification_timeline': LABEL_T,
    'evidence_status':       LABEL_E,
    'evidence_quality':      LABEL_Q,
}


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
        scores[schema_name] = sum(
            FIELD_WEIGHT[field] * f1_score(
                reference_df[field], predictions[field],
                labels=allowed, average='macro', zero_division=0,
            )
            for field, allowed in schema.items()
        )
    scores['mean'] = (scores['A'] + scores['B']) / 2
    return scores


def compute_per_field_f1(reference_df, predictions):
    return {
        field: f1_score(
            reference_df[field], predictions[field],
            labels=allowed, average='macro', zero_division=0,
        )
        for field, allowed in EVAL_SCHEMA_A.items()
    }


def read_train_val():
    part_train = pd.read_csv(os.path.join(BASE_DIR, 'vpesg_4k_train_1000.csv'), encoding='utf-8-sig')
    part_val   = pd.read_csv(os.path.join(BASE_DIR, 'vpesg4k_val_1000.csv'),   encoding='utf-8-sig')
    combined   = pd.concat([part_train, part_val], ignore_index=True)

    for col in ['verification_timeline', 'evidence_status', 'evidence_quality']:
        combined[col] = combined[col].fillna('N/A').str.strip()
    combined['promise_status'] = combined['promise_status'].str.strip()

    return combined


all_df = read_train_val()

run_scores = []
for tag in RUN_TAGS:
    logit_data  = np.load(os.path.join(BASE_DIR, f'out_{tag}', 'oof_logits.npz'))
    task_logits = {k: logit_data[k] for k in ('p', 't', 'e', 'q')}
    score_a     = compute_dual_f1(all_df, apply_label_hierarchy(task_logits))['A']
    run_scores.append(score_a)

run_scores = np.array(run_scores)
print('Seed-noise floor (timefeat-large OOF-A):')
for tag, score_a in zip(RUN_TAGS, run_scores):
    print(f'  {tag:<16} A {score_a:.4f}')
print(f'  mean {run_scores.mean():.4f} | sd {run_scores.std(ddof=1):.4f} | range {run_scores.max() - run_scores.min():.4f}')

oof_accum = {k: 0.0 for k in ('p', 't', 'e', 'q')}
for tag in RUN_TAGS:
    logit_data = np.load(os.path.join(BASE_DIR, f'out_{tag}', 'oof_logits.npz'))
    for k in oof_accum:
        oof_accum[k] = oof_accum[k] + softmax(logit_data[k], axis=1) / NUM_RUNS

oof_preds  = apply_label_hierarchy(oof_accum)
oof_score  = compute_dual_f1(all_df, oof_preds)
field_f1   = compute_per_field_f1(all_df, oof_preds)

print(f"\n4-seed ENSEMBLE OOF: A {oof_score['A']:.4f} | B {oof_score['B']:.4f} | mean {oof_score['mean']:.4f}")
print(f"  per-field(A): prom {field_f1['promise_status']:.3f}  time {field_f1['verification_timeline']:.3f}  evid {field_f1['evidence_status']:.3f}  qual {field_f1['evidence_quality']:.3f}")
print(f"  vs best single seed A {run_scores.max():.4f}  ->  delta {oof_score['A'] - run_scores.max():+.4f}  (seed-mean {run_scores.mean():.4f} -> {oof_score['A'] - run_scores.mean():+.4f})")

test_df    = pd.read_csv(os.path.join(BASE_DIR, 'vpesg4k_test_2000.csv'), encoding='utf-8-sig')
test_accum = {k: 0.0 for k in ('p', 't', 'e', 'q')}
for tag in RUN_TAGS:
    prob_data = np.load(os.path.join(BASE_DIR, f'out_{tag}', 'test_probs.npz'))
    for k in test_accum:
        test_accum[k] = test_accum[k] + prob_data[k] / NUM_RUNS

submission = pd.DataFrame({'id': test_df['id'], **apply_label_hierarchy(test_accum)})
submission = submission[['id', 'promise_status', 'verification_timeline', 'evidence_status', 'evidence_quality']]

assert len(submission) == 2000 and list(submission['id']) == list(range(12001, 14001))
assert submission.notna().all().all()
no_promise = submission['promise_status'] == 'No'
assert (submission.loc[no_promise, ['verification_timeline', 'evidence_status', 'evidence_quality']] == 'N/A').all().all()
assert (submission.loc[submission['evidence_status'] == 'No', 'evidence_quality'] == 'N/A').all()

out_path = os.path.join(BASE_DIR, 'submission_4seed_tf.csv')
submission.to_csv(out_path, index=False, encoding='utf-8')
print(f'\nsubmission written: {out_path}  (format-validated)')

anchor_path = os.path.join(BASE_DIR, 'submission_3seed_tf.csv')
if os.path.exists(anchor_path):
    anchor_df    = pd.read_csv(anchor_path, encoding='utf-8', keep_default_na=False)
    changed_rows = sum(
        (anchor_df[col].values != submission[col].values).sum()
        for col in ['promise_status', 'verification_timeline', 'evidence_status', 'evidence_quality']
    )
    print(f'  rows changed vs banked 3-seed anchor: {changed_rows}')
