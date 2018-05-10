### function to make synthetic images

import matplotlib.pyplot as plt
import numpy as np  
import random
import os
import pickle
import scipy.misc
from skimage.filters import threshold_otsu

gt_path = '/data/DATASETS/WORKABLE/Dicom_Samples/Set2/Gurpreet_Experiments/ground_truths/'

img_dir = '/data/DATASETS/WORKABLE/Dicom_Samples/Set2/Cluster_1/'

used_list = '/data/gabriel/Doppler/used_names_types.pkl'

#view_coord = np.array([0,216],dtype=np.float64)


def return_rng_params(lab_type,img_path,gt_path,used_files_pkl,inp_img):
	### returns the template image inpainted with either of the label types  
	### This function uses the output of previouis label type to inpaint new label type
	### First find a random image from the image main directory

	if lab_type == 'W':
		used_files = pickle.load(open('for_W.pkl','rb'))
	else:
		used_files = pickle.load(open(used_files_pkl,'rb'))[lab_type]

	#if(lab_type=='W'):


	if (lab_type == 'W'):
		# if it is a waveform, find col and row reshape factor
		# this value is between 0.2 and 1.2
		col_res=1.0
		#col_res = random.random()*(1) + 0.5 
		row_res = np.random.normal(loc=0.6,scale=0.05)

	elif(lab_type == 'E'):
		# only change the height of the ekg
		col_res = 1.0
		row_res = random.random()*(1) + 0.8

	else:
		# no reshapes needed for anything else
		col_res = 1.0
		row_res = 1.0

	img_list = os.listdir(img_path)
	img_list2 = [i[:i.find('.jpeg')] for i in img_list]
	random.shuffle(img_list)
	
	while(True):
		#img_name = img_list[0]
		#img_name = img_name[:img_name.find('.jpeg')]
		#print(img_name)
		#print(used_files[0])
		#break
		intersec = list(set(img_list2).intersection(set(used_files)))
		#print(intersec)
		if len(intersec)>0:
			random.shuffle(intersec)
			img_name=intersec[0]
			break
	#print(img_name)

	label = np.loadtxt(gt_path+'/'+lab_type+'/'+img_name+'_1_2_Doppler_03282018_1.0.npy')
	orig_size = plt.imread(img_path+'/'+'/'+img_name+'.jpeg').shape
	res_r = inp_img.shape[0]/orig_size[0]
	res_c = inp_img.shape[1]/orig_size[1]

	# print(res_r)
	# print(res_c)
	# exit()

	r,c,r_l,c_l = label[0,0],label[0,1],np.abs(label[2,0]-label[0,0]),np.abs(label[2,1]-label[0,1]) ## note that the labels are un normalized so the coords and the dimensins are absolute

	true_img = plt.imread(img_path+'/'+img_name+'.jpeg')

	if ('V' not in lab_type and 'W' not in lab_type):
		roi = true_img[max(0,int(r)-5):min(int(r+r_l+8),inp_img.shape[0]),int(c):min(int(c+c_l)+2,inp_img.shape[1])]
		#roi_rz = scipy.misc.imresize(roi,(int(res_r*roi.shape[0]),int(res_c*roi.shape[1]))) 
	
	elif('V' in lab_type and 'W' not in lab_type):
		roi = true_img[max(0,int(r)-10):int(r+r_l+2+3),max(int(c)-5,0):int(c+c_l)+10]

	elif('W' in lab_type):
		## Correction in waveform to include wave noise 
		#if (os.path.isfile(gt_path+'/'+'E'+'/'+img_name+'_1_2_Doppler_03282018_1.0.npy')):
		E_lab = np.loadtxt(gt_path+'/'+'E'+'/'+img_name+'_1_2_Doppler_03282018_1.0.npy')

		new_end_point = int(E_lab[0,0] - 10)

		roi = true_img[max(0,int(r)-5):new_end_point,int(c):min(int(c+c_l)+40,inp_img.shape[1])] ## This is the waveform extended towards ekg
		
		mask_region = true_img[max(0,int(r)-5):new_end_point,int(c):min(int(c+c_l-5),inp_img.shape[1])]

		
		# print(r_l)
		# print(c_l)

		W_lab = np.loadtxt(gt_path+'/'+'W'+'/'+img_name+'_1_2_Doppler_03282018_1.0.npy')
		
	#the resized image here - 
	roi_rz = scipy.misc.imresize(roi,(int(inp_img.shape[0]*row_res),int(inp_img.shape[1]*col_res)))

	#plt.imshow(roi_rz),plt.show()
	### the value of the sizes for the new image
	#exit()
	if(lab_type == 'W'):
		#mask_region = scipy.misc.imresize(roi,(int(r_l*row_res),int(c_l*col_res)))
		new_r_l = roi_rz.shape[0]	
		new_c_l = roi_rz.shape[1]
		#plt.imshow(roi_rz),plt.show()

		#inp_img = scipy.misc.imresize(temp_img,(temp_img.shape[0],max(inp_img.shape[1],new_c_l)))
		### TODO NEEDS FIXING
		print()
		new_mask_r_l = int(new_r_l) #(because it will be reshaped the same way)
		#new_mask_c_l = int(mask_region.shape[1]*col_res*res_c)
		new_mask_c_l = int(mask_region.shape[1]*col_res)
		

		new_coords = np.array([random.randint(150,inp_img.shape[0]-roi_rz.shape[0]),c]).astype(np.uint32)
		## This step inpaints the waveform
		print(new_coords[1]+roi_rz.shape[1])
		print(new_coords[0]+roi_rz.shape[0])
		inp_img[new_coords[0]:np.uint32(new_coords[0]+roi_rz.shape[0]),new_coords[1]:np.uint32(new_coords[1]+roi_rz.shape[1]),:] = roi_rz[:,:,:3]
		#return inp_img
		#new_r_l = 
		#coords_mask = np.where(W_mask>0)

		#plt.imshow(inp_img),plt.show()
		#exit()
		#print(new_c_l)
		#print((roi.shape[1]-40)*(c_l*col_res))
		new_coords = [new_coords[0],new_coords[1],new_r_l,new_mask_c_l]
		new_coords = [int(i) for i in new_coords]
		#mas = np.zeros_like(temp_img)[:,:,0]
		
		#mas[new_coords[0]:new_coords[0]+new_mask_r_l,new_coords[1]:new_coords[1]+new_mask_c_l] = 255.0
		
	elif lab_type == 'E':
		#roi = scipy.misc.imresize(roi,(roi.shape[0],inp_img.shape[1]))
		roi = scipy.misc.imresize(roi,(int(res_r*roi.shape[0]),int(res_c*roi.shape[1])))
		roi_gray = 0.299*roi[:,:,0]+0.587*roi[:,:,1]+0.114*roi[:,:,2]

		thresh = threshold_otsu(roi_gray)
		#plt.imshow(roi_gray),plt.show()
		roi_t = np.array(roi_gray > thresh)

		# print(np.where(roi_t>0))
		# plt.imshow(roi_t),plt.show()
		#print(roi_t[roi_t>0])
		new_coords = np.array([list(i) for i in np.where(roi_t>0)])
		new_coords = new_coords.T
	
		colors = [list(roi[int(i[0]),int(i[1]),:]) for i in new_coords]

		# for i in new_coords:
		# 	print(inp_img[i[0],i[1],:])
		# exit()

#		colors = []

		#print(colors)
		#exit()

		max_row = new_coords[:,0].max()
		
		#print(new_coords[0,:])
		while(True):
			vertical_adj = random.randint(0,inp_img.shape[0])
			
			if (np.max(new_coords[:,0]-vertical_adj + inp_img.shape[0])>255 and np.max(new_coords[:,0]-vertical_adj + inp_img.shape[0]) < inp_img.shape[0]):
				#break
				new_coords[:,0]= new_coords[:,0] - vertical_adj +inp_img.shape[0]-1
				break
		#print(new_coords[0,:])

		#print(len(new_coords))
		for i in range(0,len(new_coords)):
			inp_img[new_coords[i,0],new_coords[i,1],0] = colors[i][0]
			inp_img[new_coords[i,0],new_coords[i,1],1] = colors[i][1]
			inp_img[new_coords[i,0],new_coords[i,1],2] = colors[i][2]
			
			#print(colors[i][:3])
		#plt.imshow(inp_img),plt.show()

		#return inp_img
		#print(new_coords[:,0].min())
		#print(new_coords[:,1].min())
		#print(new_coords[:,0].max())
		#print(new_coords[:,1].max())
		#exit()
		new_coords = [new_coords[:,0].min(),new_coords[:,1].min(),new_coords[:,0].max()-new_coords[:,0].min(),new_coords[:,1].max()-new_coords[:,1].min()]
		#print(new_coords)
	elif lab_type == 'T':
		
		### 'T' will appear 40% of the times instead of all the time.

		if(np.random.normal(0.5,0.05)>=0.5):

			roi_rz = scipy.misc.imresize(roi_rz,(int(res_r*roi.shape[0]),int(res_c*roi.shape[1]))) 

			pos_place = [[0,0],[0,inp_img.shape[1] - roi_rz.shape[1]],[inp_img.shape[0] - roi_rz.shape[0],inp_img.shape[1] - roi_rz.shape[1]],[inp_img.shape[0] - roi_rz.shape[0],0]]
			#pos_place =[[inp_img.shape[0] - roi_rz.shape[0],inp_img.shape[1] - roi_rz.shape[1]]]
			random.shuffle(pos_place)
			new_coords = pos_place[0]
			#plt.imshow(roi_rz),plt.show()

			# print(int(roi_rz.shape[0]))
			# print(int(roi_rz.shape[1]))
			# print('new_sz ',inp_img.shape)
			# print('new_coords ',new_coords)
			# print(int(new_coords[0])+int(roi_rz.shape[0]))
			# print(int(new_coords[1])+int(roi_rz.shape[1]))
			inp_img[int(new_coords[0]):int(new_coords[0])+int(roi_rz.shape[0]),int(new_coords[1]):int(new_coords[1])+int(roi_rz.shape[1]),:] = roi_rz
			#plt.imshow(inp_img),plt.show()
			new_coords = [new_coords[0],new_coords[1],int(roi_rz.shape[0]),int(roi_rz.shape[1])]
		else:
			flag=0
		#return inp_img

	elif lab_type == 'V':
		roi_rz = scipy.misc.imresize(roi_rz,(int(res_r*roi.shape[0]),int(res_c*roi.shape[1]))) 
		#print('roi = ',roi_rz.shape)
		pos_row = random.randint(0,5)
		pos_col = random.randint(100,inp_img.shape[1]-roi_rz.shape[1]-100)
		inp_img[pos_row:pos_row+int(roi_rz.shape[0]),pos_col:pos_col+int(roi_rz.shape[1])] = roi_rz
		#plt.imshow(inp_img),plt.show()
		#return inp_img
		new_coords = [pos_row,pos_col,int(roi_rz.shape[0]),int(roi_rz.shape[1])]
		
	elif lab_type == 'C' :
		
		roi_rz = scipy.misc.imresize(roi_rz,(int(res_r*roi.shape[0]),int(res_c*roi.shape[1]))) 

		pos_row = random.randint(0,inp_img.shape[0]-roi_rz.shape[0]-100)
		pos_col = [5,inp_img.shape[1]-5-roi_rz.shape[1]]
		random.shuffle(pos_col)
		new_coords = [pos_row,pos_col[0]]
		#print(roi_rz.shape)
		inp_img[pos_row:pos_row+int(roi_rz.shape[0]),pos_col[0]:pos_col[0]+int(roi_rz.shape[1])] = roi_rz
		new_coords = [pos_row,pos_col[0],int(roi_rz.shape[0]),int(roi_rz.shape[1])]

	#elif lab_type == 'S':

	if(flag>0):
		new_coords2 = np.zeros((4,2))
		new_coords2[0,0] = new_coords[0]
		new_coords2[0,1] = new_coords[1]
		new_coords2[1,0] = new_coords[0]
		new_coords2[1,1] = new_coords[1]+new_coords[3]
		new_coords2[2,0] = new_coords[0]+new_coords[2]
		new_coords2[2,1] = new_coords2[1,1]
		new_coords2[3,0] = new_coords2[2,0]
		new_coords2[3,1] = new_coords2[0,1]
		#print(new_coords2) 
		return inp_img,new_coords2
	else:
		return inp_img,[0,0,0,0]


	# plt.imshow(roi),plt.show()
	# plt.imshow(roi_rz),plt.show()
	# print(r_l,c_l)
	# print(r_l*row_res,c_l*(col_res))
	# print()



	#print(gt_info)

	#plt.imshow(true_img),plt.show()
	#return inp_img
def is_overlap(t1,t2):
	#print(t1)
	#print(t2)
	if(t1[0]<=t2[1] and t1[1]>=t2[0]):
		return True
	return False


def corr_overlap(coords_list):

	dic = {'W':coords_list[0],'E':coords_list[1],'T':coords_list[2],'C':coords_list[3],'V':coords_list[4]}
	#print(dic['E'])
	
	#for i in ['E']:
	for i in ['W','E']:
		for j in list(dic.keys())[2:]:
			#print(list(dic.keys()))
			int1 = [ [dic[i][0,0],dic[i][2,0]],
						[dic[i][0,1],dic[i][1,1]] ]

			
			int2 = [[dic[j][0,0],dic[j][2,0]],
					[dic[j][0,1],dic[j][1,1]] ]
			#print(i)
			#print(j)
			
			if(is_overlap(int1[0],int2[0]) and is_overlap(int1[1],int2[1])):
				
				l = int1[0][1]-int1[0][0]
				if ( (-int1[0][0]+int2[0][1] > 0.6*l or -int2[0][0]+int1[0][1] > 0.6*l and i =='W') or (-int1[0][0]+int2[0][1] > 0.93*l or -int2[0][0]+int1[0][1] > 0.93*l and i =='E')) :
					
					if(int2[1][1]>=int1[1][0] and int2[1][0] <= int1[1][0]):#and int2[1][0]<= int1[1][0]):
						new_col_min = int2[1][1]
						new_col_max = int1[1][1]
					elif(int2[1][0]>=int1[1][0] and int2[1][1]>= int2[1][1]):
						new_col_min = int1[1][0]
						new_col_max = int2[1][0]
						
					# elif(int2[1][0]>=int1[1][0] and int2[1][1]<= int1[1][1]):
					# 	if(int2[1][0]-int1[1][0] >= int1[1][1]-int2[1][1]):
					# 		new_col_min = 
					# 		new_col_max = 
					# 	else:
					# 		new_col_min = 
					# 		new_col_max = 
				
					new_row_min = dic[i][0,0]
					new_row_max = dic[i][2,0]
									
					dic[i] = np.array([[new_row_min,new_col_min],[new_row_min,new_col_max],[new_row_max,new_col_max],[new_row_max,new_col_min]])

	coords_list_new = [dic['W'],dic['E'],dic['T'],dic['C'],dic['V']]

	return coords_list_new

for count in range(0,20):
	#try:
	coord_list =[]
	temp_img = np.zeros((480,640,3))
	#print(count)
	#print(count)
	for i in ['W','E','C','V','T']:
	#for i in ['E']:
		if not os.path.isdir('new_images/GT4/'+i):
			os.makedirs('new_images/GT4/'+i)

		temp_img,coords = return_rng_params(i,img_dir,gt_path,used_list,temp_img)
		#plt.imshow(temp_img),plt.show()
		#print('sz')
		#break
		#print(i)
		#print(coords)
		coords = coords.astype(np.int64)
		#if(i == 'E'):
			# print('try')
			# print(count)
			# print(i)
			# print(coords)
			# print('try')
		coord_list.append(coords)
		#gt_image = np.zeros_like(temp_img)
		#print(coords)
		#gt_image[coords[0,0]:coords[2,0],coords[0,1]:coords[1,1]]=255

		#scipy.misc.imsave('./new_images/GT3/'+i+'/GT_'+str(count)+'.png',gt_image)
		#np.savetxt('./new_images/GT3/'+i+'/GT_'+str(count)+'.npy',np.array([coords[0],coords[0]+coords[2],coords[1],coords[1]+coords[3]]))
		#print(temp_img.shape)
		#print('sz')
	
	#print(coords)
	# print('coord_list \n')
	# print(coord_list)
	# print('\n')
	# print(count)
	# print(coord_list[1])
	coords_list_new = corr_overlap(coord_list)
	#plt.imshow(temp_img.astype(np.uint8)),plt.show()
	p=0
	#print(count)
	for k in ['W','E','T','C','V']:
	#for k in ['E']:
		
		np.savetxt('./new_images/GT4/'+k+'/synth_img_'+str(count)+'.npy',coords_list_new[p])
		gt_image = np.zeros((480,640))
		#print(coords)
		coords = coords_list_new[p]
		#print(coords)
		#print(coords)
		#print(k)
		gt_image[coords[0,0]:coords[2,0],coords[0,1]:coords[1,1]]=255

		# plt.subplot(212)
		# plt.imshow(temp_img.astype(np.uint32))
		# plt.subplot(211)
		# plt.imshow(gt_image),plt.show()
		# plt.close()

		scipy.misc.imsave('./new_images/GT4/'+k+'/synth_img_'+str(count)+'.png',gt_image)
		p+=1
	#print(coord_list)

	#corr_overlap()
	#plt.close()
	#plt.imshow(gt_image),plt.show()
	#break
	#plt.imshow(temp_img),plt.show()
	scipy.misc.imsave('./new_images/images4/synth_img_'+str(count)+'.png',temp_img)

	# except:
	# 	continue






