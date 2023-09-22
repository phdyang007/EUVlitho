import numpy as np
import csv
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

nvalidate=500
ncut=1901
#ncut=1749
nmask=512

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

model = CNNmodel()
model = model.load_from_checkpoint("model.ckpt")

model.eval()

factor=np.loadtxt('/home/tanabe/emsim/SPIE2024/data/metalVtest/factor/factor0.csv',
delimiter=',',dtype='float32')
readerlm=csv.reader(open('/home/tanabe/emsim/SPIE2024/data/metalVtest/500/inputlm.csv'))
inputlm=[row for row in readerlm]
floss=open('loss.csv','w')
floss.write("Formattype,2\nmemo1\nmemo2\n")

loss = np.zeros(ncut, dtype='float32')
for idx in range(0,nvalidate):
    mask_test = np.load('/home/tanabe/emsim/SPIE2024/data/metalVtest/500/mask/'+str(idx)+'.npy')
    amp_test=np.load('/home/tanabe/emsim/SPIE2024/data/metalVtest/500/re0/'+str(idx)+'.npy')
    mask_test=mask_test.reshape(-1,1,nmask,nmask)
    mask_test=torch.from_numpy(mask_test)
    predict_test=model(mask_test).detach().numpy()
    for i in range(0,ncut):
        loss[i]=loss[i]+np.power(predict_test[0][i]-amp_test[i]*factor[i],2)/nvalidate
for i in range(0,ncut):
    sdv=np.sqrt(loss[i])
    floss.write(inputlm[i][0]+","+inputlm[i][1]+","+str(sdv)+"\n")
floss.close()




