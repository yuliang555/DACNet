#!/usr/bin/env python3
import os
import re
import glob
import shlex
import subprocess
import argparse
import csv
import yaml
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIGS_DIR = os.path.join(ROOT, 'configs')

LOSS_LIST = ['mae', 'mse']
PRED_LEN_LIST = [96, 192, 336, 720]

parser = argparse.ArgumentParser(description='Run all experiments')
parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of the data file')
args = parser.parse_args()


def process_results():
    # 读取 results_all.csv
    with open('results_all.csv', 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    # 处理每行
    processed = []
    for row in data:
        setting = row['settings']
        mse = float(row['MSE'])
        mae = float(row['MAE'])

        # 提取 dataset: In_ 与 _ 之间的
        match = re.search(r'In_([^_]+)_', setting)
        dataset = match.group(1) if match else 'unknown'

        # 提取 horizon: pl= 与 _norm 之间的
        match = re.search(r'pl=(\d+)_norm', setting)
        horizon = match.group(1) if match else 'unknown'

        processed.append({'dataset': dataset, 'horizon': horizon, 'MSE': mse, 'MAE': mae})

    # 步骤2: 对于相同 dataset 和 horizon，取最小 MSE 和 MAE
    min_dict = defaultdict(lambda: {'MSE': float('inf'), 'MAE': float('inf')})
    for p in processed:
        key = (p['dataset'], p['horizon'])
        min_dict[key]['MSE'] = min(min_dict[key]['MSE'], p['MSE'])
        min_dict[key]['MAE'] = min(min_dict[key]['MAE'], p['MAE'])

    df_min = [{'dataset': k[0], 'horizon': k[1], 'MSE': v['MSE'], 'MAE': v['MAE']} for k, v in min_dict.items()]

    # 步骤3: 对于相同 dataset，计算 MSE 和 MAE 平均值，horizon='avg'
    avg_dict = defaultdict(list)
    for item in df_min:
        avg_dict[item['dataset']].append((item['MSE'], item['MAE']))

    df_avg = []
    for ds, values in avg_dict.items():
        mse_avg = sum(v[0] for v in values) / len(values)
        mae_avg = sum(v[1] for v in values) / len(values)
        df_avg.append({'dataset': ds, 'horizon': 'avg', 'MSE': mse_avg, 'MAE': mae_avg})

    # 合并，按 dataset 分组，使相同 dataset 的行挨在一起
    final = []
    datasets = sorted(set(k[0] for k in min_dict.keys()))
    for ds in datasets:
        # 添加该 dataset 的所有 min 行
        for item in df_min:
            if item['dataset'] == ds:
                final.append(item)
        # 添加该 dataset 的 avg 行
        for item in df_avg:
            if item['dataset'] == ds:
                final.append(item)

    # 写入 results_all_new.csv
    with open('results_all_new.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['dataset', 'horizon', 'MSE', 'MAE'])
        writer.writeheader()
        writer.writerows(final)


def parse_yml(yml_file):
    with open(yml_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data


def run_one(cfg, loss, pred_len, gpu=0):
    cmd = ['python', '-u', 'run.py']
    # set required
    cmd.extend(['--is_training', '1'])
    cmd.extend(['--loss', loss])
    cmd.extend(['--pred_len', str(pred_len)])
    cmd.extend(['--gpu', str(gpu)])
    cmd.extend(['--use_gpu', 'True'])
    # add other from cfg
    for key, value in cfg.items():
        if key in ['loss', 'pred_len', 'gpu', 'is_training']:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])
    print('=> Running:', ' '.join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    config_files = sorted(glob.glob(os.path.join(CONFIGS_DIR, '*.yml')))
    if len(config_files) == 0:
        print('No config files found in', CONFIGS_DIR)
        raise SystemExit(1)

    total = len(config_files) * len(LOSS_LIST) * len(PRED_LEN_LIST)
    done = 0

    for config_path in config_files:
        name = os.path.basename(config_path).replace('.yml', '')
        print(f"\n=== {name} ===")
        config_data = parse_yml(config_path)
        fixed = config_data['fixed']
        pred_len_configs = config_data['pred_len_configs']

        for loss in LOSS_LIST:
            for pred_len in PRED_LEN_LIST:
                if pred_len not in pred_len_configs:
                    continue
                cfg = fixed.copy()
                cfg.update(pred_len_configs[pred_len])
                cfg['pred_len'] = pred_len
                cfg['root_path'] = args.root_path  # override if needed
                cfg['run_type'] = 'all'

                done += 1
                print(f"[{done}/{total}] {name} loss={loss} pred_len={pred_len}")
                try:
                    run_one(cfg, loss, pred_len, gpu=0)
                except subprocess.CalledProcessError as e:
                    print(f"ERROR in {name} loss={loss} pred_len={pred_len}:", e)
                    raise
                
    process_results()

    print('All experiments completed.')
