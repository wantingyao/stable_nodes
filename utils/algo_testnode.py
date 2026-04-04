import os, sys
from pathlib import Path
import numpy as np
import torch
from torch import optim, nn, Tensor
import pytorch_lightning as pl
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from copy import deepcopy
import pickle
import random
import collections
import h5py
import matplotlib.pyplot as plt
import pyLasaDataset as lasa
from dataset import load_expanded_dataset_hdf5
from torchdiffeq import odeint

class testNODE(pl.LightningModule):
    def __init__(self, hyper_model: nn.Module, params: dict) -> None:
        super(testNODE, self).__init__()
        self.params = params
        self.model = hyper_model
        self.curr_device = None

    def forward(self, batch) -> Tensor:
        input = batch["pos"][:,0,:] # (bs, 1000, 2)
        ts = batch["t"].squeeze(-1)  # (bs, 1000)

        pos_pred = self.model(ts, input)

        return pos_pred

    
    def training_step(self, batch, batch_idx):
        output = batch["pos"]
        bs=output.shape[0]
        self.curr_device = batch["pos"].device
        output_pred = self.forward(batch=batch)
        train_loss = F.mse_loss(output_pred, output)
        self.log("train/train_loss", train_loss.detach(),
             prog_bar=True, on_step=True, on_epoch=True, batch_size=bs)
        return train_loss
    

    def validation_step(self, batch, batch_idx):
        output = batch["pos"]
        bs=output.shape[0]
        self.curr_device = batch["pos"].device
        with torch.no_grad():
            output_pred = self.forward(batch=batch)
            val_loss = F.mse_loss(output_pred, output)
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True, sync_dist=True, batch_size=bs)

    def on_train_epoch_end(self):
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        params = list(set(self.model.parameters()))
                 
        optimizer = optim.Adam(params, lr=self.params['LR'], weight_decay=self.params['weight_decay'])
        scheduler = {'scheduler': optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.params['max_epochs'], eta_min=self.params['LR']*0.1),}


        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
    
    
    def testing(self, cfg=None, device=torch.device("cuda")):
        self.model.eval()
        self.curr_device = device

        lasa_data = getattr(lasa.DataSet, cfg.task_name)

        dt = lasa_data.dt
        demos = lasa_data.demos

        demo_0 = demos[0]
        pos = demo_0.pos # np.ndarray, shape: (2,2000)
        vel = demo_0.vel # np.ndarray, shape: (2,2000) 
        acc = demo_0.acc # np.ndarray, shape: (2,2000)
        t = demo_0.t # np.ndarray, shape: (1,2000)

        # To visualise the data (2D position and velocity) use the plot_model utility
        # lasa.utilities.plot_model(lasa_data) # give any of the available 
                                                        # pattern data as argument
        ndemos = len(demos)
        T = demos[0].t.shape[-1]
        pos_all = []
        vel_all = []
        for i in range(ndemos):
            pos_all.append((demos[i].pos).T)
            vel_all.append((demos[i].vel).T)
        posn = np.array(pos_all)
        ts =  np.array(t.T).reshape(T)

        train_indx = 0
        plt.plot(posn[train_indx, :, 0], posn[train_indx, :, 1], c="dodgerblue", label="Real")
        plt.plot(posn[train_indx, 0, 0], posn[train_indx, 0, 1], c="black", marker='o', markersize = '12', label="Start")
        plt.plot(posn[train_indx, -1, 0], posn[train_indx, -1, 1], c="black", marker='x', markersize = '12', label="Target")
        
        with torch.no_grad():
            model_y = self.model.inference(ts, posn[train_indx, 0])
        plt.plot(model_y[:, 0], model_y[:, 1], c="crimson", label="Model")
        with torch.no_grad():
            model_y = self.model.inference(ts, posn[train_indx, 0] + np.array([0, 2]))
        plt.plot(model_y[:, 0], model_y[:, 1], c="crimson", label="Model")
        with torch.no_grad():
            model_y = self.model.inference(ts, posn[train_indx, 0] + np.array([2, 0]))
        plt.plot(model_y[:, 0], model_y[:, 1], c="crimson", label="Model")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{cfg.eval_path}_neural_ode.png")
        plt.show()
        print("Evaluation plot saved at: ", f"{cfg.eval_path}_neural_ode.png")