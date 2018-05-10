import torch
import numpy as np 
import matplotlib.pyplot as plt 
from networks import *
from data import *

from torchvision import models


label_list = ['W','E','V','C','T']



#label_dir1 = '/data/DATASETS/WORKABLE/Dicom_Samples/Set2/Gurpreet_Experiments/ground_truths/'
#img_dir = '/data/DATASETS/WORKABLE/Dicom_Samples/Set2/Cluster_1/'

label_dir1 = '/data/gabriel/Doppler/new_images/GT2/'
img_dir = '/data/gabriel/Doppler/new_images/images2/'

model_dirs = {'dense161':'/data/gabriel/Doppler/nndev/models/densenet161_model_embedsz.pth.tar','res101':'/data/gabriel/Doppler/nndev/models/resnet101_model_embedsz.pth.tar'
					,'res18':'/data/gabriel/Doppler/nndev/models/resnet18_model_embedsz.pth.tar'}


for fe_dirs in list(model_dirs.keys())[-1:]:
	for label_name in label_list[:3]:
		
		if(label_name in 'WEV'):
			gray_param = True
			data_add = 'gray'
		else:
			gray_param = False
			data_add = ''
		label_dir = label_dir1 + label_name

		layers = 1
		hd = 10000
		for d in [0.2,0.5]:
			model_name = '2model_'+fe_dirs+'_l_'+str(layers)+'_hd_'+str(hd)+'_d_'+str(d)+'_type_'+label_name+'_synth'
			network = dop_bbox(fe_obj=None,mlp_l = layers,mlp_hd = hd,dr=d,model_dir = model_dirs[fe_dirs])

			# m = train(model=network,epochs=300,batch_size=8,data_dir=img_dir,label_dir=label_dir,lr=0.005,steps=[100,200],lr_mult=0.8
			# 	,model_name=model_name,label_type=label_name,name_list_dir='/data/gabriel/Doppler/used_names_types.pkl',save_dir='./data_loaders/',norm_bool=True)
			# print(label_name)
			# print(model_dirs)
			# print(d)
			#if not(label_name=='W' and fe_dirs=='res18' and d == 0.2):
			m = train(model=network,epochs=250,batch_size=8,data_dir=img_dir,label_dir=label_dir,lr=0.001,steps=[100],lr_mult=0.5
				,model_name=model_name,label_type=label_name,save_dir='./data_loaders/',norm_bool=True,data_name='synth_2_res_480640' + label_name+data_add,gray=gray_param)

			

			torch.save(m,'./results/'+model_name+'/'+model_name+'synth.pth')
			m = torch.load('./results/'+model_name+'/'+model_name+'synth.pth')
			test(m,save_dir='./data_loaders/',data_name='synth_2_res_480640' + label_name+data_add,model_name=model_name)
			#break
		#break
	#break

