import torch
from torchvision import models,transforms
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os

class dop_bbox(nn.Module):
	def __init__(self,fe_obj,mlp_l,mlp_hd,dr,model_dir=None):
		super(dop_bbox,self).__init__()

		self.fe = nn.Sequential()
		count=0
		self.model_dir = model_dir
		if(self.model_dir == None):
			for j,i in fe_obj.named_children():
				
				#print(i)
				if isinstance(i,nn.Linear):
					
					self.embed_sz = i.in_features
					break
				self.fe.add_module('fe_'+str(count),i)
				count+=1
		else:

			saved_obj =  torch.load(self.model_dir)
			self.embed_sz = saved_obj['embed_sz']
			self.fe = saved_obj['model']
		self.classifier = nn.Sequential()
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		#self.avgpool = nn.AvgPool2d(7,stride=1)
		if(mlp_l==1):
			self.classifier.add_module('dense_0',nn.Linear(self.embed_sz,4))
			self.classifier.add_module('sigmoid_0',nn.Sigmoid())
			#self.classifier.add_module('dense_0_act',nn.ReLU())
		else:
			self.classifier.add_module('dense_0',nn.Linear(self.embed_sz,mlp_hd))
			#self.classifier.add_module('dense_0_act',nn.ReLU())
			
			for i in range(1,mlp_l-1):
				#if(i==mlp_l):
				self.classifier.add_module('dense_'+str(i),nn.Linear(mlp_hd,mlp_hd))
				self.classifier.add_module('dropout_'+str(i),nn.Dropout(p=dr,inplace=True))
				#self.classifier.add_module('dense_'+str(i)+'_act',nn.ReLU())

			self.classifier.add_module('classifier_',nn.Linear(mlp_hd,4))
			self.classifier.add_module('sigmoid_',nn.Linear(mlp_hd,4))

			#self.classifier.add_module('classifier_',)



	def forward(self,x):
		#print(self.fe)
		x = self.fe(x)

		if('densenet' in self.model_dir):
			x = self.avgpool(x)
		#print('output_of fe')
		#print(x.size())
		#x = self.avgpool(x)
		#print(x.size())
		x = x.view(x.size(0),-1)
		#print('output_of fe2')
		#print(x.size())
		x = self.classifier(x)
		#print('output')
		#print(x.size())
		return x



class netD(nn.Module):
	def __init__(self,gpu=1):
		super(netD,self).__init__()
		self.gpu = gpu

		# self.network = nn.Sequential(
		# 	nn.Conv2d(3,64,4,2,1,bias=False),
		# 	nn.LeakyReLU(0.2,inplace=True),
		# 	nn.Conv2d(64,128,4,2,1,bias=False),
		# 	nn.BatchNorm2d(128),
		# 	nn.LeakyReLU(0.2,inplace=True),
		# 	nn.Conv2d(128,256,4,2,1,bias=False),
		# 	nn.BatchNorm2d(256),
		# 	nn.LeakyReLU(0.2,inplace=True),
		# 	nn.Conv2d(256,512,4,2,1,bias=False),
		# 	nn.BatchNorm2d(512),
		# 	nn.LeakyReLU(0.2,inplace=True),
		# 	nn.Conv2d(512,1,4,1,0,bias=False),
		# 	nn.Sigmoid()
		# 	)

		self.network = nn.Sequential(
			nn.Conv2d(3,64,4,2,1,bias=False),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv2d(64,128,4,2,1,bias=False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv2d(128,256,4,2,1,bias=False),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv2d(256,512,4,2,1,bias=False),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv2d(512,1024,4,2,1,bias=False),
			nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.2,inplace=True),
			nn.Conv2d(1024,2048,4,2,1,bias=False),
			nn.BatchNorm2d(2048),
			nn.LeakyReLU(0.2),

			nn.Conv2d(2048,1,4,1,0,bias=False),
			nn.Sigmoid()
			)
	def forward(self,x):
		if(self.gpu>=1):
			output=nn.parallel.data_parallel(self.network,x,range(self.gpu))
		else:
			output = self.network(x)

		return output.view(-1,1).squeeze(1)

class netG(nn.Module):
	def __init__(self,gpu=1,z=100):
		super(netG,self).__init__()
		self.gpu=gpu
		self.z=z
		# self.network= nn.Sequential(
		# 	nn.ConvTranspose2d(z,64*8,4,1,0,bias=False),
		# 	nn.BatchNorm2d(64*8),
		# 	nn.ReLU(inplace=True),

		# 	nn.ConvTranspose2d(64*8,64*4,4,2,1,bias=False),
		# 	nn.BatchNorm2d(64*4),
		# 	nn.ReLU(inplace=True),

		# 	nn.ConvTranspose2d(64*4,64*2,4,2,1,bias=False),
		# 	nn.BatchNorm2d(64*2),
		# 	nn.ReLU(inplace=True),

		# 	nn.ConvTranspose2d(64*2,64,4,2,1,bias=False),
		# 	nn.BatchNorm2d(64),
		# 	nn.ReLU(inplace=True),

		# 	nn.ConvTranspose2d(64,3,4,2,1,bias=False),
		# 	nn.Tanh()

		# 	)

		self.network= nn.Sequential(

			
			nn.ConvTranspose2d(self.z,64*32,4,1,0,bias=False),
			nn.BatchNorm2d(64*32),
			nn.ReLU(inplace=True),

			nn.ConvTranspose2d(64*32,64*16,4,2,1,bias=False),
			nn.BatchNorm2d(64*16),
			nn.ReLU(inplace=True),

			nn.ConvTranspose2d(64*16,64*8,4,2,1,bias=False),
			nn.BatchNorm2d(64*8),
			nn.ReLU(inplace=True),

			nn.ConvTranspose2d(64*8,64*4,4,2,1,bias=False),
			nn.BatchNorm2d(64*4),
			nn.ReLU(inplace=True),

			nn.ConvTranspose2d(64*4,64*2,4,2,1,bias=False),
			nn.BatchNorm2d(64*2),
			nn.ReLU(inplace=True),
			
			nn.ConvTranspose2d(64*2,64*1,4,2,1,bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),

			nn.ConvTranspose2d(64,3,4,2,1,bias=False),
			nn.Tanh()

			)


	def forward(self,x):
		if self.gpu>=1:
			output = nn.parallel.data_parallel(self.network,x,range(self.gpu))
		else:
			output = self.network(x)

		return output
