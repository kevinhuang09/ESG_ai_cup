import subprocess
import os
shared_env = os.environ.copy()
shared_env['ESG_MODEL'] = 'IDEA-CCNL/Erlangshen-Roberta-330M-Similarity'
shared_env['ESG_TIMEFEAT'] = '1'
shared_env['ESG_GRADCKPT'] = '1'
shared_env['ESG_CLASSWEIGHT'] = '1'
shared_env['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
shared_env['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
experiments = [{'TAG': 'cv_els_tf', 'SEED': '42'}, {'TAG': 'cv_els_tf_s1', 'SEED': '1'}, {'TAG': 'cv_els_tf_s2', 'SEED': '2'}, {'TAG': 'cv_els_tf_s3', 'SEED': '3'}]
print('🚀 開始執行【二郎神二號主力模型】自動化訓練腳本...')
for i, exp in enumerate(experiments, 1):
    current_env = shared_env.copy()
    current_env['ESG_TAG'] = exp['TAG']
    current_env['ESG_SEED'] = exp['SEED']
    print('\n' + '=' * 50)
    print(f' 正在執行進度 [{i}/{len(experiments)}]')
    print(f" 🏷️ TAG  : {exp['TAG']}")
    print(f" 🌱 SEED : {exp['SEED']}")
    print(f" 🤖 MODEL: {shared_env['ESG_MODEL']}")
    print('=' * 50 + '\n')
    result = subprocess.run(['python', 'train.py'], env=current_env)
    if result.returncode != 0:
        print(f"❌ 錯誤：實驗 {exp['TAG']} 執行失敗，程式中斷。")
        break
else:
    print('\n🎉【二郎神全部實驗執行完畢！】🎉')
    print('現在你可以用這四個 Seed 的 test_probs.npz 去跟 RoBERTa 做集成囉！')