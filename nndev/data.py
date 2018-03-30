from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from PIL import Image
from torch import nn
import torch.optim

from torch.optim import lr_scheduler
def load_data(data_dir,label_dir,batch_size):
	### TODO , wait for the data to be selected and then write this function
	# yield image_batch,label_batch (probably corner x, corner y, length, breadth)
	raise NotImplementedError


def train(model,epochs,batch_size,data_dir,label_dir,lr,steps,lr_mult,model_name='model_100hd_1l'):
	### data_dir = /bla/bla/bla/containing_train_and_test_folders

	data_loaders = {'val':load_data(data_dir=data_dir+'/val/',label_dir=label_dir,batch_size=batch_size),
					'train':load_data(data_dir=data_dir+'/train/',label_dir=label_dir,batch_size=batch_size)}

	criterion = nn.MSELoss()
	opt = torch.optim.SGD(model.parameters(),lr=lr )
	lr_scheduler_model = lr_scheduler.MultiStepLR(optimizer=opt,milestones = steps,gamma=lr_mult)
	
	mse_epoch_val = []
	mse_epoch_train = []
		
	for epoch in range(epochs):
		running_loss_val = 0
		running_loss_train = 0

		print(epoch)

		for phase in ['train','val']:

			if phase == 'train':
				model.train(True)
			else:
				model.train(False)

			for img,label in data_loaders[phase]:

				if(phase=='train'):

					img_var = Variable(img.cuda())
					label_var = Variable(label.cuda())

				else:
					img_var = Variable(img.cuda(),volatile=True)
					label_var = Variable(label.cuda(),volatile=True)

				output = model(img_var)

				loss = criterion(output,label_var)

				if(phase='train'):
					loss.backward()
					lr_scheduler_model.step()
					mse_epoch_train.append(loss.cpu().data.numpy()[0])
				else:
					mse_epoch_val.append(loss.cpu().data.numpy()[0])


		print('train_mse = ',mse_loss_train[-1])
		print('val_mse = ',mse_loss_val[-1])

		plt.plot(mse_epoch_val)
		plt.savefig('./val_loss_plot_'+model_name+'.png')
		plt.close()
		plt.plot(mse_epoch_train)
		plt.savefig('./train_loss_plot_'+model_name+'.png')
		plt.close()

