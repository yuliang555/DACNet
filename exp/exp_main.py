from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import DACNet_In, DACNet_Out 
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import pandas
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import torch.nn.functional as F

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np
import psutil

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'DACNet_In': DACNet_In,
            'DACNet_Out': DACNet_Out,
        }
        map_raw = self._cyclemap()             
        model = model_dict[self.args.model].Model(self.args, map_raw).float()
        print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _cyclemap(self):
        train_data = self._get_data('train')[0].data_x

        if any(substr in self.args.model for substr in {'Out'}):
            self.args.intra_len = self.args.cycle + self.args.pred_len
        else:
            self.args.intra_len = self.args.cycle + self.args.seq_len

        train_data = torch.tensor(train_data).permute(1, 0).float().to(self.device)
        map_raw = train_data.unfold(dimension=1, size=self.args.intra_len, step=self.args.cycle).permute(0, 2, 1)    # (C, P+L, S)
        self.args.inter_len = map_raw.shape[2]
        print(map_raw.shape)     
                
        return map_raw

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'mse':            
            criterion = nn.MSELoss()
        elif self.args.loss == 'mae':
            criterion = nn.L1Loss()
        elif self.args.loss == 'huber':
            criterion = nn.SmoothL1Loss(beta=self.args.beta)        
        return criterion

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_indices) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_indices = batch_indices.long().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, batch_indices)
                else:
                    outputs = self.model(batch_x, batch_x_mark, batch_indices)

                pred = outputs.detach().cpu()
                true = batch_y
                loss = criterion(pred, true)                                
                total_loss.append(loss)
                
        total_loss = np.average(total_loss)
        self.model.train()
        
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_indices) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_indices = batch_indices.long().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, batch_indices)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, batch_indices)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                # if (i + 1) % 100 == 0:
                #     print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                #     speed = (time.time() - time_now) / iter_count
                #     left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                #     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                #     iter_count = 0
                #     time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
       
        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        
        result_path = f'./Figure/result/{self.args.model_id}/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_indices) in enumerate(test_loader):
                batch_y = batch_y.float()
                batch_x = batch_x.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_indices = batch_indices.long().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, batch_indices)
                else:
                    outputs = self.model(batch_x, batch_x_mark, batch_indices)

                pred = outputs.detach().cpu().numpy()
                true = batch_y.numpy()

                preds.append(pred)
                trues.append(true)           

        if self.args.test_flop:
            test_params_flop(self.model, (batch_x.shape[1], batch_x.shape[2]))
            exit()
            
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        # inputx = np.concatenate(inputx, axis=0)

        # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        # inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        # result save
        # folder_path = './results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        mae, mse = metric(preds, trues)

        metric_path = f"Print/{self.args.model_id}_{self.args.pred_len}.csv"
        if not os.path.exists(metric_path):
            open(metric_path, "w").close()                      
        df = pandas.DataFrame({
            "settings": [f"{setting}"],
            "MSE": [mse],
            "MAE": [mae],
        })
        if os.path.getsize(metric_path) == 0:
            df.to_csv(metric_path, index=None)
        else:
            df_old = pandas.read_csv(metric_path)
            df_new = pandas.concat([df_old, df], ignore_index=True)
            df_new.to_csv(metric_path, index=None) 
                    
        print('mse:{}, mae:{}'.format(mse, mae))
        
        # f = open(f"result_{self.args.pred_len}.txt", 'a')
        # f.write(setting + "  \n")
        # f.write('mse:{}, mae:{}'.format(mse, mae))
        # f.write('\n')
        # f.write('\n')
        # f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    # def predict(self, setting, load=False):
    #     pred_data, pred_loader = self._get_data(flag='pred')

    #     if load:
    #         path = os.path.join(self.args.checkpoints, setting)
    #         best_model_path = path + '/' + 'checkpoint.pth'
    #         self.model.load_state_dict(torch.load(best_model_path))

    #     preds = []

    #     self.model.eval()
    #     with torch.no_grad():
    #         for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_cycle) in enumerate(pred_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_y = batch_y.float()
    #             batch_x_mark = batch_x_mark.float().to(self.device)
    #             batch_y_mark = batch_y_mark.float().to(self.device)
    #             batch_cycle = batch_cycle.int().to(self.device)

    #             # decoder input
    #             dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float().to(
    #                 batch_y.device)
    #             dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
    #             # encoder - decoder
    #             if self.args.use_amp:
    #                 with torch.cuda.amp.autocast():
    #                     if any(substr in self.args.model for substr in {'Cycle'}):
    #                         outputs = self.model(batch_x, batch_cycle)
    #                     elif any(substr in self.args.model for substr in
    #                              {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF'}):
    #                         outputs = self.model(batch_x)
    #                     else:
    #                         if self.args.output_attention:
    #                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                         else:
    #                             outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #             else:
    #                 if any(substr in self.args.model for substr in {'Cycle'}):
    #                     outputs = self.model(batch_x, batch_cycle)
    #                 elif any(substr in self.args.model for substr in {'Linear', 'MLP', 'SegRNN', 'TST', 'SparseTSF'}):
    #                     outputs = self.model(batch_x)
    #                 else:
    #                     if self.args.output_attention:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
    #                     else:
    #                         outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
    #             pred = outputs.detach().cpu().numpy()  # .squeeze()
    #             preds.append(pred)

    #     preds = np.array(preds)
    #     preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

    #     # result save
    #     folder_path = './results/' + setting + '/'
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    #     np.save(folder_path + 'real_prediction.npy', preds)

    #     return
