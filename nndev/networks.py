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
		if(model_dir == None):
			for j,i in fe_obj.named_children():
				
				#print(i)
				if isinstance(i,nn.Linear):
					
					self.embed_sz = i.in_features
					break
				self.fe.add_module('fe_'+str(count),i)
				count+=1
		else:

			self.fe = torch.load(model_dir)
		self.classifier = nn.Sequential()
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		#self.avgpool = nn.AvgPool2d(7,stride=1)
		if(mlp_l==1):
			self.classifier.add_module('dense_0',nn.Linear(self.embed_sz,4))
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




