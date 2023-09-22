import matplotlib.pyplot as plt
import numpy as np
import csv
import gc
import os
import torch
from torch import nn
import random
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import torchvision
from torchvision.transforms import functional as Fv
import pandas as pd

ndata=800000
nvalidate=500
nsample=800000

ncut=1901
#ncut=1749
nmask=512

factor=np.loadtxt('/home/tanabe/emsim/SPIE2024/data/metalVtest/factor/factor0.csv',
delimiter=',',dtype='float32')

class MaskampDatasetTrain(Dataset):
    def __init__(self, list_IDs):
        self.list_IDs = list_IDs
    def __len__(self):
        return len(self.list_IDs)
    def __getitem__(self, idx):
        ID=self.list_IDs[idx]
        x=np.load('/home/tanabe/emsim/SPIE2024/data/metalVtest/800000/mask/'+str(ID)+'.npy')
        x=x.reshape(1,nmask,nmask)
        x=torch.from_numpy(x)
        y=np.load('/home/tanabe/emsim/SPIE2024/data/metalVtest/800000/re0/'+str(ID)+'.npy')
        y=y*factor
        y=torch.from_numpy(y)
        return x, y

class MaskampDatasetVal(Dataset):
    def __len__(self):
        return nvalidate
    def __getitem__(self, idx):
        x=np.load('/home/tanabe/emsim/SPIE2024/data/metalVtest/500/mask/'+str(idx)+'.npy')
        x=x.reshape(1,nmask,nmask)
        x=torch.from_numpy(x)
        y=np.load('/home/tanabe/emsim/SPIE2024/data/metalVtest/500/re0/'+str(idx)+'.npy')
        y=y*factor
        y=torch.from_numpy(y)
        return x, y

class CNNmodel(pl.LightningModule):
	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1,32,(3,3),padding=1,padding_mode='circular')
		self.conv2 = nn.Conv2d(32,64,(3,3),padding=1,padding_mode='circular')
		self.conv3 = nn.Conv2d(64,128,(3,3),padding=1,padding_mode='circular')
		self.conv4 = nn.Conv2d(128,256,(3,3),padding=1,padding_mode='circular')
		self.conv5 = nn.Conv2d(256,512,(3,3),padding=1,padding_mode='circular')
		self.conv6 = nn.Conv2d(512,1024,(3,3),padding=1,padding_mode='circular')
		self.norm1=nn.BatchNorm2d(32)
		self.norm2=nn.BatchNorm2d(64)
		self.norm3=nn.BatchNorm2d(128)
		self.norm4=nn.BatchNorm2d(256)
		self.norm5=nn.BatchNorm2d(512)
		self.norm6=nn.BatchNorm2d(1024)
		self.pool = nn.MaxPool2d(2, 2)
		self.fc1 = nn.Linear(8*8*1024, 2048)
		self.fc2=nn.Linear(2048, ncut)
	def forward(self, x):
		x=x.to(torch.float32)
		x = self.pool(F.relu(self.conv1(x)))
		x=self.norm1(x)
		x = self.pool(F.relu(self.conv2(x)))
		x=self.norm2(x)
		x = self.pool(F.relu(self.conv3(x)))
		x=self.norm3(x)
		x = self.pool(F.relu(self.conv4(x)))
		x=self.norm4(x)
		x = self.pool(F.relu(self.conv5(x)))
		x=self.norm5(x)
		x = self.pool(F.relu(self.conv6(x)))
		x=self.norm6(x)
		x=torch.flatten(x, 1)
		x = F.relu(self.fc1(x))
		outputs = self.fc2(x)
		return outputs
	def configure_optimizers(self):
#		optimizer = torch.optim.Adam(self.parameters())
#		optimizer = torch.optim.SGD(self.parameters(),lr=1.,momentum=0.)
#		optimizer = torch.optim.SGD(self.parameters(),lr=2.,momentum=0.)
		optimizer = torch.optim.SGD(self.parameters(),lr=0.5,momentum=0.)
		return optimizer
	def training_step(self, train_batch, batch_idx):
		mask, targets= train_batch
		preds = self.forward(mask)
		loss = F.mse_loss(preds, targets)	
		self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
		return loss
	def validation_step(self, val_batch, batch_idx):
		mask, targets = val_batch
		preds = self.forward(mask)
		loss = F.mse_loss(preds, targets)
		self.log('val_loss', loss)
		for i in range(len(preds[0,:])):
			loss = F.mse_loss(preds[:,i], targets[:,i])
			self.log('val_loss_'+str(i), loss, sync_dist=True)

def training():
   list_IDs=random.sample(range(nsample),ndata)
   data_set_of_train = MaskampDatasetTrain(list_IDs)
   data_set_of_val = MaskampDatasetVal()

   train_dataloader=DataLoader(data_set_of_train,batch_size=128,shuffle=True,num_workers=min(16,os.cpu_count()))
   val_dataloader=DataLoader(data_set_of_val,batch_size=128,num_workers=min(16,os.cpu_count()))
#   train_dataloader=DataLoader(data_set_of_train,batch_size=256,shuffle=True,num_workers=8)
#   val_dataloader=DataLoader(data_set_of_val,batch_size=256,num_workers=8)

   model = CNNmodel()
   logger = pl.loggers.CSVLogger("./",version=0, name="history")
   trainer = pl.Trainer(accelerator='gpu',devices=[0,1,2,3],strategy="ddp",logger=logger,max_epochs=100)
#   trainer = pl.Trainer(accelerator='gpu',devices=[0,1,2,3],strategy="ddp",logger=logger,max_epochs=100)
#   trainer = pl.Trainer(accelerator='gpu',devices=[0],logger=logger,max_epochs=2)
   trainer.fit(model, train_dataloader, val_dataloader)
#   trainer.save_checkpoint("best_model.ckpt", weights_only=True)

   trainer.save_checkpoint("model.ckpt")
#   model = model.load_from_checkpoint("model.ckpt")
#   torch.save(model.state_dict(), 'model.pt')
#model = CNNmodel.load_from_checkpoint("best_model.ckpt")

   log=pd.read_csv('./history/version_0/metrics.csv')
   log=log.groupby("epoch").max()
   log.to_csv('./log.csv')

   loss_values=log['train_loss']
   val_loss_values=log['val_loss']
   epochs=range(1,len(loss_values)+1)
   plt.rcParams.update({'font.size': 16})
   plt.plot(epochs,loss_values,'bo',label='Training loss')
   plt.plot(epochs,val_loss_values,'r',label='Validation loss')
   plt.ylim(bottom=0)
   plt.xlabel('Epochs')
   plt.ylabel('Loss')
   plt.legend()
   plt.tight_layout()
   plt.savefig('./out.png')
   plt.show()

if __name__ == "__main__":
   training()

