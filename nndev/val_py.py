### testing on unused val cohort

import torch
from torch.autograd import Variable
from PIL import Image
from networks import *
import matplotlib.patches as patches
import os

val_path = '/data/DATASETS/WORKABLE/Dicom_Samples/Val_Set_04092018/'
#model_path = '/data/gabriel/Doppler/nndev/results/model_l_1_hd_7500_d_0_type_E/model_l_1_hd_7500_d_0_type_E.pth'
#model_path = '/data/gabriel/Doppler/nndev/results/model_dense161_l_1_hd_10000_d_0.2_type_W_synth/model_dense161_l_1_hd_10000_d_0.2_type_W_synthsynth.pth'
#model = torch.load(model_path).eval().cuda()

data_tf = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])

model_list = ['model_res18_l_1_hd_10000_d_0.2_type_E_synth', 'model_res18_l_1_hd_10000_d_0.5_type_E_synth', 
			'model_res18_l_1_hd_10000_d_0.2_type_W_synth', 'model_res18_l_1_hd_10000_d_0.2_type_C_synth', 
			#'model_res18_l_1_hd_10000_d_0.5_type_C_synth', 
			'model_res18_l_1_hd_10000_d_0.5_type_W_synth', 
			'model_res18_l_1_hd_10000_d_0.2_type_V_synth', 'model_res18_l_1_hd_10000_d_0.5_type_V_synth']



for m in model_list:
	model = torch.load('results/'+'2'+m+'/'+'2'+m+'synth.pth').train(False).cuda()
	label_name = m[m.find('_synth')-1:]
	for i in os.listdir(val_path):
		img_pil = Image.open(val_path+'/'+i)
		img_np = plt.imread(val_path+'/'+i)
		c,r = img_pil.size
		img_tensor = Variable(data_tf(img_pil).unsqueeze(0).cuda(),volatile=True)

		out = model(img_tensor)
		#print(out)
		out_np = out.cpu().data.squeeze().numpy()

		row_out,col_out = out_np[2]*r,out_np[3]*c

		row_coord,col_coord = out_np[0]*r,out_np[1]*c

		fig,ax = plt.subplots(1)
		ax.imshow(img_np)


		rect_output= patches.Rectangle((col_coord,row_coord),col_out,row_out,linewidth=2,edgecolor='r',facecolor='none')

		ax.add_patch(rect_output)

		plt.savefig('val_results/'+'2'+m+'_'+label_name+i+'.png')

		plt.close()

				