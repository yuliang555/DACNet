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

LOSS_LIST = ['mae']
PRED_LEN_LIST = [720]
SEED_LIST = [2024, 2025, 2026, 2027, 2028]

parser = argparse.ArgumentParser(description='Run all experiments')
parser.add_argument('--root_path', type=str, default='./datasets/', help='root path of the data file')
parser.add_argument('--dataset', type=str, help='dataset name, e.g. ecl, etth1, ...')
parser.add_argument('--gpu', type=int, default=0, help='gpu id (default 0)')
args = parser.parse_args()


def parse_yml(yml_file):
    with open(yml_file, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return data


def run_one(cfg, seed, loss, pred_len, gpu=0):
    cmd = ['python', '-u', 'run.py']
    # set required
    cmd.extend(['--is_training', '1'])
    cmd.extend(['--loss', loss])
    cmd.extend(['--random_seed', str(seed)])
    cmd.extend(['--pred_len', str(pred_len)])
    cmd.extend(['--gpu', str(gpu)])
    cmd.extend(['--use_gpu', 'True'])
    # add other from cfg
    for key, value in cfg.items():
        if key in ['seed','loss', 'pred_len', 'gpu', 'is_training']:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(f'--{key}')
        else:
            cmd.extend([f'--{key}', str(value)])
    print('=> Running:', ' '.join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__':
    config_file = os.path.join(ROOT, 'configs', f'{args.dataset.lower()}.yml')
    if not os.path.exists(config_file):
        print(f'Config file {config_file} not found')
        raise SystemExit(1)

    config_data = yaml.safe_load(open(config_file, 'r'))
    fixed = config_data.get('fixed', {})
    pred_len_configs = config_data.get('pred_len_configs', {})

    for seed in SEED_LIST:
        for loss in LOSS_LIST:
            for pred_len in PRED_LEN_LIST:
                if pred_len not in pred_len_configs:
                    continue
                cfg = fixed.copy()
                cfg.update(pred_len_configs[pred_len])
                cfg['pred_len'] = pred_len
                cfg['root_path'] = args.root_path  # override if needed
                cfg['run_type'] = 'single'

                run_one(cfg, seed, loss, pred_len, gpu=args.gpu)


    print('Single experiments completed.')
