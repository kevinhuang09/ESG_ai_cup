import subprocess
import os
shared_env = os.environ.copy()
shared_env['ESG_MODEL'] = 'BAAI/bge-large-zh-v1.5'
shared_env['ESG_TIMEFEAT'] = '1'
shared_env['ESG_GRADCKPT'] = '1'
shared_env['ESG_MAXLEN'] = '512'
shared_env['ESG_CLASSWEIGHT'] = '1'
shared_env['ESG_SPANAUX'] = '1'
experiments = [{'TAG': 'bgeL_tf', 'SEED': '42', 'EPOCHS': '8', 'LR': '1e-5'}, {'TAG': 'bgeL_tf1', 'SEED': '42', 'EPOCHS': '10', 'LR': '1e-5'}, {'TAG': 'bgeL_tf2', 'SEED': '42', 'EPOCHS': '8', 'LR': '2e-5'}, {'TAG': 'bgeL_tf3', 'SEED': '42', 'EPOCHS': '10', 'LR': '2e-5'}]
print('🚀 開始執行 BGE-Large 終極壓榨效能自動化訓練...')
for i, exp in enumerate(experiments, 1):
    current_env = shared_env.copy()
    current_env['ESG_TAG'] = exp['TAG']
    current_env['ESG_SEED'] = exp['SEED']
    current_env['ESG_EPOCHS'] = exp['EPOCHS']
    current_env['ESG_LR'] = exp['LR']
    print('\n' + '=' * 50)
    print(f' 正在執行進度 [{i}/{len(experiments)}]')
    print(f" 🏷️ TAG    : {exp['TAG']}")
    print(f" 📅 EPOCHS : {exp['EPOCHS']}  |  Learning Rate: {exp['LR']}")
    print(f" 🌱 SEED   : {exp['SEED']}")
    print('=' * 50 + '\n')
    result = subprocess.run(['python', 'train.py'], env=current_env)
    if result.returncode != 0:
        print(f"❌ 錯誤：實驗 {exp['TAG']} 執行失敗，程式中斷。")
        break
else:
    print('\n🎉【優化組合實驗全部執行完畢！】🎉')