from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from PIL import Image
from torch import nn
import torch.optim
import torch.utils.data as data_utils
from torch.optim import lr_scheduler
import random
import pickle
import matplotlib.patches as patches
from PIL import Image
def load_data(data_dir,label_dir,batch_size,name_list_dir,label_type,save_dir,model_name,ow=True):
	### TODO , wait for the data to be selected and then write this function
	# yield image_batch,label_batch (probably corner x, corner y, length, breadth)

	#zoom=str(zoom)

	if not os.path.isdir(save_dir) or ow:

		name_list = pickle.load(open(name_list_dir,'rb'))[label_type]

		label_end = '_1_2_Doppler_03282018_1.0.npy'

		img_tensor_list = []
		label_tensor_list = []
		size_tensor_list = []
		data_tf = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])


		count = 0

		#tensor_img = torch.zeros((1,256,256))

		for i in os.listdir(data_dir):


			name_img = i[:i.find('.jpeg')]
			if name_img in name_list:
				label_name = name_img+label_end
				img_name = name_img+'.jpeg'

				np_img = plt.imread(data_dir+'/'+img_name)
				the_img = Image.open(data_dir+'/'+img_name)

				label = np.loadtxt(label_dir+'/'+label_name)
				#print(the_img.size)
				cols,rows= the_img.size # Image_obj.size returns as cols x rows

				label_tf = np.array([label[0,0],label[0,1],np.abs(label[2,0]-label[0,0]),np.abs(label[2,1]-label[0,1]),rows,cols]).reshape(1,6).astype(np.float32)
				##### NOTE - Ground Truth arranged as ROW INDEX, COLUMN INDEX , ROW_LENGTH, COLUMN_LENGTH , TOTAL_ROWS, TOTAL_COLS
				# print(label[0,0])
				# print(label[0,1])
				# print(label_tf[0,2])
				# print(label_tf[0,3])
				# print(label_tf[0,4])
				# print(label_tf[0,5])
				# plt.close()
				# plt.imshow(the_img),plt.show()
				#print(label[0,0],label[0,1])
				#print(label[2,0],label[])	
				label_tf = torch.from_numpy(label_tf/np.array([rows,cols,rows,cols,1.0,1.0]))
				
				img_tensors = data_tf(the_img).unsqueeze(0)
				
				img_tensor_list.append(img_tensors)
				
				label_tensor_list.append(label_tf)

		all_idx = np.arange(0,len(img_tensor_list))
		random.shuffle(all_idx)
		test_idx = torch.LongTensor(all_idx[int(0.9*len(all_idx)):].astype(int))
		val_idx = torch.LongTensor(all_idx[int(0.7*len(all_idx)):int(0.9*len(all_idx))].astype(int))
		train_idx = torch.LongTensor(all_idx[:int(0.7*len(all_idx))].astype(int))

		#print(test_idx)

		img_tensor_list = torch.cat(img_tensor_list,dim=0)
		#print(img_tensor_list.size())
		label_tensor_list =torch.cat(label_tensor_list,dim=0)
		#print(label_tensor_list.size())



		train_data = data_utils.DataLoader(data_utils.TensorDataset(img_tensor_list[train_idx,:,:],label_tensor_list[train_idx,:].float()),batch_size=batch_size,shuffle=True)
		test_data = data_utils.DataLoader(data_utils.TensorDataset(img_tensor_list[test_idx,:,:],label_tensor_list[test_idx,:].float()),batch_size=batch_size,shuffle=True)
		val_data = data_utils.DataLoader(data_utils.TensorDataset(img_tensor_list[val_idx,:,:],label_tensor_list[val_idx,:].float()),batch_size=batch_size,shuffle=True)

		torch.save(train_data,save_dir+'/'+model_name+'train.pth')
		torch.save(test_data,save_dir+'/'+model_name+'test.pth')
		torch.save(val_data,save_dir+'/'+model_name+'val.pth')

	else:
		train_data = torch.load(save_dir+'/'+model_name+'train.pth')
		test_data = torch.load(save_dir+'/'+model_name+'test.pth')
		val_data = torch.load(save_dir+'/'+model_name+'val.pth')

	return {'train':train_data,'test':test_data,'val':val_data}

from torch.autograd import Variable
import scipy.misc
import os
def train(model,epochs,batch_size,data_dir,label_dir,lr,steps,lr_mult,label_type,name_list_dir,save_dir,model_name='model_100hd_1l'):
	### data_dir = /bla/bla/bla/containing_train_and_test_folders
	if not os.path.isdir('./results/'+model_name):
		os.makedirs('./results/'+model_name)
	data_loaders = load_data(data_dir = data_dir,label_dir=label_dir,batch_size=batch_size,name_list_dir=name_list_dir,label_type=label_type,save_dir=save_dir,model_name=model_name)

	criterion = nn.MSELoss()
	opt = torch.optim.SGD(model.parameters(),lr=lr )
	lr_scheduler_model = lr_scheduler.MultiStepLR(optimizer=opt,milestones = steps,gamma=lr_mult)
	
	mse_epoch_val = []
	mse_epoch_train = []
	model=model.cuda()
	for epoch in range(epochs):
		running_loss_val = 0
		running_loss_train = 0

		print(epoch)
		print(model_name)
		for phase in ['train','val']:

			if phase == 'train':
				lr_scheduler_model.step()
				model.train(True)

			else:
				model.train(False)

			for img,label in data_loaders[phase]:
				#print(label)
				label = label[:,:4]
				#print(label)
				if(phase=='train'):
					
					img_var = Variable(img.cuda())
					label_var = Variable(label.cuda())
					opt.zero_grad()

				else:
					img_var = Variable(img.cuda(),volatile=True)
					label_var = Variable(label.cuda(),volatile=True)
				
				#print(label_var)
				output = model(img_var)
				#print(output)

				loss = criterion(output,label_var)

				if(phase=='train'):
					loss.backward()
					opt.step()
					running_loss_train+=loss.cpu().data[0]
				else:
					running_loss_val+=loss.cpu().data[0]

			if phase=='train':
				mse_epoch_train.append(running_loss_train)
			else:
				mse_epoch_val.append(running_loss_val)

		print('train_mse = ',mse_epoch_train[-1])
		print('val_mse = ',mse_epoch_val[-1])

		plt.plot(mse_epoch_val)
		plt.savefig('./results/'+model_name+'/val_loss_plot_'+model_name+'.png')
		plt.close()
		plt.plot(mse_epoch_train)
		plt.savefig('./results/'+model_name+'/train_loss_plot_'+model_name+'.png')
		plt.close()

	return model

def test(model,save_dir,model_name):
	
	test_load = torch.load(save_dir+'/'+model_name+'test.pth')
	model.train(False)
	model=model.cuda()
	count=0
	for img,label in test_load:
		img = Variable(img.cuda(),volatile=True)



		actual_lab = label[:,:4].numpy()
		#### Arranged as Row normed, col normed, row_length normed, col_length normed

		sz = np.concatenate((label[:,4:],label[:,4:]),axis=1)
		#print(sz)
		output = model(img)
		output_np = output.cpu().data.numpy()
		### saving the test images
		for i in range(0,img.size(0)):
			img_np = img[i,:,:,:].cpu().data.numpy().transpose(1,2,0)

						

			img_np = scipy.misc.imresize(img_np,(sz[i,:2][0],sz[i,:2][1])).astype(np.int8)

			fig,ax = plt.subplots(1)
			ax.imshow(img_np)

			corner_output = (output_np[i,0]*sz[i,0],output_np[i,1]*sz[i,1])
			#print(corner)
			#print(actual_lab[i,1]*sz[i,1],actual_lab[i,0]*sz[i,0])
			#print(output_np[i,3])

			corner =  (actual_lab[i,1]*sz[i,1],actual_lab[i,0]*sz[i,0])

			b = actual_lab[i,2]*sz[i,0]
			l = actual_lab[i,3]*sz[i,1]

			b_output = output_np[i,2]*sz[i,0]
			l_output = output_np[i,3]*sz[i,1]
			#print(l,b)

			rect_output= patches.Rectangle(corner,l_output,b_output,linewidth=2,edgecolor='r',facecolor='none')
			rect_actual= patches.Rectangle(corner,l,b,linewidth=2,edgecolor='g',facecolor='none')
			#count+=1
			ax.add_patch(rect_output)
			ax.add_patch(rect_actual)
			count+=1
			plt.savefig('./results/'+model_name+'/'+str(count)+'.png')
			#plt.show()
			#fig.close()
			plt.close()




