#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import datetime
import os
import time
import imp
import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset
from tqdm import tqdm
import math 
import sklearn.metrics as metrics
import pandas as pd
import warnings
from datasets import preprocessingDataset
from torch.utils.data import Dataset, DataLoader
import MinkowskiEngine as ME
from torch.optim import AdamW
import torch.optim as optim
import time
import pandas as pd
from model.tenext import TENext
import numpy as np


class Trainer():
    def __init__(self, ARCH_kitti, DATA_kitti, DATA_rellis, DATA_usl, datadir, datadir2, datadir3):
        # parameters
        self.model_yaml = ARCH_kitti
        self.kitti_yaml = DATA_kitti #content yml
        self.datadir_kitti = datadir # root path kitti

        self.rellis_yaml = DATA_rellis #content yml
        self.datadir_rellis = datadir2 # root path rellis


        self.usl_yaml = DATA_usl #content yml
        self.datadir_usl = datadir3 # root path usl
        self.train_data_final, self.valid_data= self.load_data()


    def load_data(self):
        full_dataset_kitti = preprocessingDataset(self.datadir_kitti,"train",self.model_yaml["train"]["voxel_size"], "ply")
        full_dataset_rellis = preprocessingDataset(self.datadir_rellis,"train", self.model_yaml["train"]["voxel_size"], "ply")
        datasets_train = ConcatDataset([full_dataset_rellis, full_dataset_kitti])
        train_data_final = DataLoader(datasets_train, batch_size=self.model_yaml["train"]["batch_size"], collate_fn=ME.utils.batch_sparse_collate,num_workers=16,shuffle=True)
        valid_data_kitti_rellis = preprocessingDataset("/media/arvc/HDD4TB1/Antonio/Minkowski/dataset_recortado_normales_vecinos/valid-refined","valid", self.model_yaml["train"]["voxel_size"])
        valid_data_final = DataLoader(valid_data_kitti_rellis, batch_size=self.model_yaml["train"]["batch_size"], collate_fn=ME.utils.batch_sparse_collate,num_workers=self.model_yaml["train"]["num_workers"],shuffle=True)
        print("Training with ", len(full_dataset_kitti), " samples of SemanticKITTI")
        print("Training with ", len(full_dataset_rellis), " samples of Rellis-3D")
        print("Testing with ", len(valid_data_kitti_rellis), " samples of Rellis-3D and SemanticKITTI")
        return train_data_final,valid_data_final
 

    def train(self): #training function
        best_f1=0.0
        best_th=0.0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.BCELoss()
        self.model = TENext(1, 1).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Number of parameters: {total_params}")
        self.optimizer = AdamW(self.model.parameters(), lr=self.model_yaml["train"]["lr"], weight_decay=self.model_yaml["train"]["weight_decay"], eps=1e-8)#normalmente es 1e-1 
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=int(len(self.train_data_final)/30),T_mult=1, eta_min=0)
        self.iters = len(self.train_data_final)
        for epoch in range(0, self.model_yaml["train"]["max_epochs"]):
            loss = self.train_epoch(train_data=self.train_data_final,
                                    model=self.model,
                                    criterion=self.criterion,
                                    optimizer=self.optimizer,
                                    epoch=epoch,
                                    scheduler=self.scheduler,
                                    device=self.device,
                                    iters=self.iters)
        #llamar al compute th y al test que toque y guardar el modelo y terminas
            best_th=self.compute_th(val_loader=self.valid_data, model=self.model, criterion=self.criterion, device=self.device)
            mean_p,mean_r,mean_f1=self.test(val_loader=self.valid_data, model=self.model, criterion=self.criterion, mean_th=best_th,device=self.device)
            if mean_f1 > best_f1:
                best_metric = mean_f1
                model_def = self.model
                th_def = best_th
            else:
                h = h + 1
                print("No se guarda este modelo, ya que no mejora lo anterior", epoch)
                print("------------------")
            if h==15:
                torch.save(model_def.state_dict(),
                'model/BestModel'+str(epoch)+'_th_'+str(th_def)+"voxel_size_"+str(mean_f1)+'.pth')
                print("------------------")
                break
        print('Finished Training')

            


    def train_epoch(self, train_data, model, criterion, optimizer, epoch, scheduler,device,iters):
        self.model.train(mode=True)
        for i,data in enumerate(tqdm(self.train_data_final)):
            coords, feats, label= data
            optimizer.zero_grad()
            in_field = ME.TensorField(feats.to(dtype=torch.float32),coordinates=coords,quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                        minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,device=device)
            # Forward
            input = in_field.sparse()
            output = model(input) 
            out_field = output.slice(in_field)
            # Loss
            loss = criterion(out_field.F, label.to(device).unsqueeze(1).float()) #.F son las features, esta funcion es una clase abstracta para calcular el gradiente
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + i / iters)
        return 0
        
    def compute_th(self, val_loader, model, criterion, device):
        model.eval()
        optimal_th_list = []
        torch.cuda.empty_cache()

        with torch.no_grad():
            end = time.time()
            for i, cloud in enumerate(tqdm(val_loader)):
                test_coords, test_feats, test_label = cloud
                test_in_field = ME.TensorField(test_feats.to(dtype=torch.float32),
                                      coordinates=test_coords,
                                      quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                      minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, device=device)

                test_output = model(test_in_field.sparse())
                logit = test_output.slice(test_in_field)
                test_label_gt = test_label.cpu().numpy()
                precision, recall, thresholds = metrics.precision_recall_curve(test_label_gt, logit.F.cpu().numpy())
                fscore = (2 * precision * recall) / (precision + recall)
                # locate the index of the largest f score
                ix = np.argmax(fscore)
                if math.isnan(fscore[ix]):
                    pass
                else:
                    print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
                    if thresholds[ix]==0:
                        pass
                    else:
                        optimal_th_list.append(thresholds[ix])

        return sum(optimal_th_list)/len(optimal_th_list)


    def test(self, val_loader, model, criterion, mean_th, device):
        # net.to(device)
        # torch.cuda.set_device(device)
        model.eval()
        all_accuracy = []
        all_recall = []
        all_precision = []
        print("Calculando métricas")
        with torch.no_grad():
            for i, cloud in enumerate(tqdm(val_loader)):
                test_coords, test_feats, test_label = cloud

                test_in_field = ME.TensorField(test_feats.to(dtype=torch.float32),
                                        coordinates=test_coords,
                                        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                        minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED, device=device)

                test_label = test_label.to(device)
                test_output = model(test_in_field.sparse())
                logit = test_output.slice(test_in_field)
                val_loss = criterion(logit.F, test_label.unsqueeze(1).float())
                test_label_gt = test_label.cpu().numpy()
                pred=np.where(logit.F.cpu().numpy() > mean_th, 1, 0)
                all_accuracy.append(metrics.accuracy_score(pred, test_label_gt))
                all_recall.append(metrics.recall_score(test_label_gt, pred))
                all_precision.append(metrics.precision_score(test_label_gt, pred))
                print('\t\t Loss:', val_loss.item())

            mean_r=sum(all_recall) / len(all_recall)
            mean_p=sum(all_precision) / len(all_precision)
            mean_f1 = 2 * (mean_p * mean_r) / (mean_p + mean_r)
            print('Mean Precision all batches of validation:', mean_p, '\t Threshold:', mean_th)
            print('Mean Recall all batches of validation:', mean_r, '\t Threshold:', mean_th)

        return mean_p,mean_r,mean_f1