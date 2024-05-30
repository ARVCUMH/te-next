import sklearn.metrics
import torch
import torch.nn as nn
from torch.optim import SGD
import MinkowskiEngine as ME
from torch.utils.data import Dataset, DataLoader
import glob
from tqdm import tqdm
import numpy as np
import open3d as o3d
import os
import sys
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
from plyfile import PlyData
import pandas as pd
import time
from model.tenext import TENext
from torch.utils.data import Dataset, DataLoader
from datasets import preprocessingDataset
import yaml
if __name__ == '__main__':

    with open("config/TE.yaml") as stream:
        try:
            config=yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    models2evaluate=[config['test']['model_path']] #path to the model
    th = [0.4862739620804787]
    datasets2evaluate=[config['test']['test_path']] #path to the dataset
    voxel = [0.2]
    dataset = []
    model_name = []
    accuracy_final = []
    F1_final = []
    recall_final = []
    precision_final = []
    MIOU_final = []
    p_w_final=[]
    r_w_final=[]
    time_final = []
    tn_final = []
    fp_final = []
    fn_final = []
    tp_final = []
    j=0

    for x, l in enumerate(models2evaluate):
        model = TENext(1 , 1).to(device)
        model.load_state_dict(torch.load(l))
        criterion = nn.BCELoss()
        for d in datasets2evaluate:
            test_dataset = preprocessingDataset(root_data=d, mode="valid",voxel=voxel[x])  # este valid es un test en realidad
            test_data = DataLoader(test_dataset, batch_size=1, collate_fn=ME.utils.batch_sparse_collate, num_workers=1)

            all_accuracy = []
            all_f1score = []
            all_recall = []
            all_precision = []
            all_miou = []
            all_p_w = []
            all_r_w = []
            all_time = []
            tn_all=0
            fp_all=0
            fn_all=0
            tp_all=0
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Number of parameters: {total_params}")
            for i, data in enumerate(tqdm(test_data)):
                coords, features, label = data
                coords=coords.to(device)
                features = features.to(device)
                label = label.to(device)

                test_in_field = ME.TensorField((features).to(dtype=torch.float32),
                                            coordinates=(coords),
                                            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                                            minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                                            device=device)
                start_time = time.time()

                test_output = model(test_in_field.sparse())
                all_time.append(time.time() - start_time)
                logit = test_output.slice(test_in_field)
                val_loss = criterion(logit.F, label.unsqueeze(1).float())
                test_label_gt = label.cpu().numpy()
                pred_raw = logit.F.detach().cpu().numpy()
                pred = np.where(pred_raw > th[x], 1, 0)

                all_accuracy.append(metrics.accuracy_score(test_label_gt, pred))
                all_f1score.append(metrics.f1_score(test_label_gt, pred))
                all_recall.append(metrics.recall_score(test_label_gt, pred))
                all_precision.append(metrics.precision_score(test_label_gt, pred))
                all_miou.append(metrics.jaccard_score(test_label_gt, pred))
                all_p_w.append(metrics.precision_score(test_label_gt, pred, average="weighted"))
                all_r_w.append(metrics.recall_score(test_label_gt, pred, average="weighted"))
                c_m = metrics.confusion_matrix(test_label_gt, pred)
                tn_all=tn_all+c_m[0,0]
                fp_all=fp_all+c_m[0,1]
                fn_all=fn_all+c_m[1,0]
                tp_all=tp_all+c_m[1,1]

            dataset.append(dataset)
            accuracy_final.append(sum(all_accuracy) / len(all_accuracy))
            F1_final.append(sum(all_f1score) / len(all_f1score))
            recall_final.append(sum(all_recall) / len(all_recall))
            precision_final.append(sum(all_precision) / len(all_precision))
            MIOU_final.append(sum(all_miou) / len(all_miou))
            p_w_final.append(sum(all_p_w) / len(all_p_w))
            r_w_final.append(sum(all_r_w) / len(all_r_w))
            time_final.append(sum(all_time) / len(all_time))
            tn_final.append(tn_all)
            fp_final.append(fp_all)
            fn_final.append(fn_all)
            tp_final.append(tp_all)


    df = pd.DataFrame(list(zip(dataset, precision_final,recall_final,F1_final,p_w_final,r_w_final,accuracy_final,MIOU_final, time_final, tn_final, fp_final, fn_final, tp_final)),
               columns =["Dataset","precision", "recall", "F1","P_w","R_w","accuracy","MIOU","time_final","tn", "fp", "fn", "tp"])
    df.to_csv(config['test']['output_csv'])