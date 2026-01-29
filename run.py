import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np
import subprocess

parser = argparse.ArgumentParser(description='Model family for Time Series Forecasting')

# random seed
parser.add_argument('--random_seed', type=int, default=2025, help='random seed')

# basic config
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, default='test', help='model id')
parser.add_argument('--model', type=str, default='DACNet_In',
                    help='model name, options: [DACNet_In, DACNet_Out]')

# data loader
parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./datasets', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')  #fixed
parser.add_argument('--pred_len', type=int, default=720, help='prediction sequence length')

# DACNet
parser.add_argument('--use_norm', type=int, default=1, help='0: no norm, 1: norm_type_1, 2: norm_type_2')
parser.add_argument('--D_cp', type=int, default=16, help='compression dimension')
parser.add_argument('--D_de', type=int, default=16, help='denoising dimension')
parser.add_argument('--D_mix', type=int, default=8, help='mixing dimension')
parser.add_argument('--beta', type=float, default=0.1, help='Huber Loss beta')
parser.add_argument('--sim_mode', type=str, default='l1', help="simlarity measure: ['l1', 'l2', 'cosine', 'dot', 'pearson']")
parser.add_argument('--mix', type=int, default=1, help="0: no mixing, 1: with mixing")
parser.add_argument('--backbone', type=str, default='mlp', help="options: ['linear', 'dlinear', 'mlp', 'dmlp', 'itransformer']")
parser.add_argument('--cycle', type=int, default=96, help='period of data')

# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder', default=True)
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

# random seed
fix_seed = args.random_seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
    
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
                            
        setting = '{}_{}_{}_{}_d={}_lr={}_lradj={}_pl={}_norm={}_mix={}_Dcp={}_Dde={}_Dmix={}_sim={}'.format(            
            args.model,
            args.model_id,                               
            args.backbone,
            args.loss,                            
            args.d_model,
            args.learning_rate,
            args.lradj,                       
            args.pred_len,
            args.use_norm,
            args.mix,
            args.D_cp,
            args.D_de,
            args.D_mix,
            args.sim_mode
        )                            

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        torch.cuda.empty_cache()
else:

    setting = '{}_{}_{}_{}_d={}_lr={}_lradj={}_pl={}_norm={}_mix={}_Dcp={}_Dde={}_Dmix={}_sim={}'.format(            
        args.model,
        args.model_id,                               
        args.backbone,
        args.loss,                            
        args.d_model,
        args.learning_rate,
        args.lradj,                       
        args.pred_len,
        args.use_norm,
        args.mix,
        args.D_cp,
        args.D_de,
        args.D_mix,
        args.sim_mode
    )

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()
