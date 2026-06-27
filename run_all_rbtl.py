import subprocess
import os
shared_env = os.environ.copy()
shared_env['ESG_MODEL'] = 'hfl/chinese-roberta-wwm-ext-large'
shared_env['ESG_TIMEFEAT'] = '1'
shared_env['ESG_GRADCKPT'] = '1'
experiments = [{'TAG': 'cv_rbtL_tf', 'SEED': '2'}, {'TAG': 'cv_rbtL_tf_s1', 'SEED': '5'}, {'TAG': 'cv_rbtL_tf_s2', 'SEED': '888'}, {'TAG': 'cv_rbtL_tf_s3', 'SEED': '6'}]
print('🚀 開始執行自動化訓練腳本...')
for i, exp in enumerate(experiments, 1):
    current_env = shared_env.copy()
    current_env['ESG_TAG'] = exp['TAG']
    current_env['ESG_SEED'] = exp['SEED']
    print('\n' + '=' * 50)
    print(f' 正在執行進度 [{i}/{len(experiments)}]')
    print(f" 🏷️ TAG  : {exp['TAG']}")
    print(f" 🌱 SEED : {exp['SEED']}")
    print('=' * 50 + '\n')
    result = subprocess.run(['python', 'train.py'], env=current_env)
    if result.returncode != 0:
        print(f"❌ 錯誤：實驗 {exp['TAG']} 執行失敗，程式中斷。")
        break
else:
    print('\n🎉【全部實驗執行完畢！】🎉')