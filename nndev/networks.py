import torch
from torchvision import models,transforms
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import os

class dop_bbox(nn.Module):
	def __init__(self,fe_obj,mlp_l,mlp_hd):
		super(dop_bbox,self).__init__()

		self.fe = nn.Sequential()
		count=0
		for i in fe_obj.children():
			if not isinstance(i,nn.Linear):
				self.fe.add_module('fe_'+str(count),i)
			else:
				self.embed_sz = i.in_features
				break
		self.classifier = nn.Sequential()

		if(mlp_l==1):
			self.classifier.add_module('dense_0',nn.Linear(self.embed_sz,4))
		else:
			self.classifier.add_module('dense_0',nn.Linear(self.embed_sz,mlp_hd))

			for i in range(1,mlp_l-1):
				#if(i==mlp_l):
				self.classifier.add_module('dense_'+str(i),nn.Linear(mlp_hd,mlp_hd))

			self.classifier.add_module('classifier_',nn.Linear(mlp_hd,4))

	def forward(self,x):
		x = self.fe(x)
		x = x.view(x.size(0),-1)
		x = self.classifier(x)
		return x




