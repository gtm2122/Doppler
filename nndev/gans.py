### GANS training

import torch
from torch import nn
from networks import netD,netG
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import torchvision.utils as vutils
def load_data_gan(data_dir,label_dir,save_dir,label_name = None,ow = False,batch_size=8):

	data_tf = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
	count = 0

	true_label = 1
	start_point = 0
	end_point=batch_size
	
	img_names = os.listdir(data_dir)

	for start in range(0,len(os.listdir(data_dir))-len(os.listdir(data_dir))%batch_size,batch_size):
		count= 0
		for img_name in img_names[start:start+batch_size]:
			
			
			#print(label)
			img_part = Image.open(data_dir+'/'+img_name)

			img_npy = np.asarray(img_part,dtype='int32')

			if label_name != None:

				label = np.loadtxt(label_dir+'/'+img_name.replace('.png','')+'.npy').astype(np.uint32)

				img_wave_part = img_npy[label[0,0]:label[2,0],label[0,1]:label[1,1],:]
			else:
				img_wave_part = img_npy
			#print(img_wave_part.shape)
			img_image = Image.fromarray(np.asarray(img_wave_part,dtype="uint8"),"RGB")
			#img_image_g = Image.fromarray(np.asarray(img_wave_part[:,:,1],dtype="uint8"),"L")
			#img_image_b = Image.fromarray(np.asarray(img_wave_part[:,:,2],dtype="uint8"),"L")



			#scipy.misc.imsave('./temp.png',img_image) 
			#exit()

			#print(img_part[0:5,0:5,2])
			#exit()
			img_tf = data_tf(img_image).unsqueeze(0)
			#print(img_tf.size())
			#plt.imshow(img_tf.numpy().squeeze().transpose(1,2,0)),plt.show()
			#exit()
			if(count==0):
				img_tensor = img_tf
				count+=1
				#label_tensor = torch.FloatTensor().fill_(true_label)
			else:
				img_tensor = torch.cat((img_tensor,img_tf),dim=0)
				#label_tensor = torch.cat((label_tensor,torch.FloatTensor().fill_(true_label)))

		start+=batch_size
		#print(start)
		#print(img_tensor.size())
		yield img_tensor
		#exit()

	
def weights_init(m):
	classname = m.__class__.__name__
	if(classname.find('Conv') !=-1):
		m.weight.data.normal_(0,0.02)
	elif(classname.find('BatchNorm')!=-1):
		m.weight.data.normal_(1,0.02)
		m.bias.data.fill_(0)

from torch import optim
from torch.autograd import Variable
def train_gans(epochs,batch_sz = 16,lr=0.002,latent_vec_sz = 256):

	D = netD(2).cuda()
	G = netG(2,latent_vec_sz).cuda()

	D.apply(weights_init)

	criterion = nn.BCELoss()

	optimizerD = optim.Adam(D.parameters(),lr = lr,betas=(0.5,0.999))
	optimizerG = optim.Adam(G.parameters(),lr = lr,betas=(0.5,0.999))
 	
	fixed_noise = torch.FloatTensor(batch_sz,latent_vec_sz,1,1).normal_(0,1)

	noise_input = torch.FloatTensor(batch_sz,latent_vec_sz,1,1)
	
	fixed_noise  = Variable(fixed_noise.cuda())
	epoch_lossG=[]
	epoch_lossD=[]
	for epoch in range(0,epochs):
		im_count = 0
		print('epoch = ',epoch)
		running_loss_D = 0
		running_loss_G = 0
		for img_batch in load_data_gan(data_dir='/data/gabriel/Doppler/new_images/images2/',label_dir='/data/gabriel/Doppler/new_images/GT2/W/',
			label_name='W',save_dir='/data/gabriel/Doppler/nndev/gan_dataset/',ow = True,batch_size=batch_sz):

			#print('im_count= ',im_count)
			# real training
			D.zero_grad()

			img_batch_v = Variable(img_batch.cuda())
			#print(img_batch_v)
			true_label = Variable(torch.FloatTensor(batch_sz).fill_(1.).cuda())	

			out_D = D(img_batch_v)

			loss_D1 = criterion(out_D,true_label)

			loss_D1.backward()
			#fake img training
			#print(noise_input.size())
			
			noise_input.normal_(0.0,1.0)
			#print(noise_input.size())
			fake_img = G(Variable(noise_input.cuda()))

			D_output_fake = D(fake_img.detach())
			
			fake_labels = Variable(torch.FloatTensor(batch_sz).fill_(0).cuda())

			loss_D2 = criterion(D_output_fake,fake_labels)

			loss_D2.backward()

			optimizerD.step()

			total_d_loss = loss_D2 + loss_D1
			running_loss_D += total_d_loss.cpu().data
			#G training

			G.zero_grad()

			label_g = Variable(torch.FloatTensor(batch_sz).fill_(1.).cuda())

			loss_G = criterion(D(fake_img),label_g)

			running_loss_G += loss_G.cpu().data

			loss_G.backward()

			optimizerG.step()

			im_count+=1

		#print(running_loss_G)
		#print(running_loss_G[0])
		
		epoch_lossD.append(running_loss_D[0])
		epoch_lossG.append(running_loss_G[0])

		plt.plot(epoch_lossD)
		plt.savefig('gan_losses/D.png')
		plt.close()
		plt.plot(epoch_lossG)
		plt.savefig('gan_losses/lossG.png')
		plt.close()

		if epoch > 10 :
			torch.save(D,'saved_models_gans/Dmodel_checkpoint_'+str(epoch)+'.pth')
			torch.save(G,'saved_models_gans/Gmodel_checkpoint_'+str(epoch)+'.pth')

		#if(im_count%100==0):
		fake = G(fixed_noise)

		#vutils.save_image(img_batch,'%s/real_samples_epoch_%03d.png'%('/data/gabriel/Doppler/nndev/gan_results/',epoch),normalize=True)

		vutils.save_image(fake.cpu().data,'%s/fake_samples_epoch_%03d.png'%('/data/gabriel/Doppler/nndev/gan_results/',epoch),normalize=True)

print(train_gans(epochs=30,latent_vec_sz=100,batch_sz=8))


# for i in a:

# 	print(i)


#def train_gan():
