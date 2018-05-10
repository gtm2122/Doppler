import matplotlib.pyplot as plt 
import numpy as np 

import pandas as pd

a = pd.read_excel('/data/DATASETS/WORKABLE/Dicom_Samples/Set2/Gurpreet_Experiments/combined.xlsx')

dop_dir = '/data/DATASETS/WORKABLE/Dicom_Samples/Set2/Gurpreet_Experiments/DATE03282018/'
orig_dir = '/data/DATASETS/WORKABLE/Dicom_Samples/Set2/Cluster_1/'

zoom = '1.0'

#print(np.array(a))

import os

import scipy.io
import scipy.misc
#for i in os.listdir(dop_dir):

#print(a[0])

#print(a['Wcolor'])
import re

t = 'Imgtype'
c = 'color'

import os
from PIL import Image
import matplotlib.patches as patches

gt_path = '/data/DATASETS/WORKABLE/Dicom_Samples/Set2/Gurpreet_Experiments/ground_truths/'

bbox_types = ['W','E','V','C','T','S']
#bbox_types = ['W','V','T']

c_to_num = {'b':0,'r':1,'g':2,'y':3,'m':4}

f = open('./error_names.txt','w')
f.close()
ccc = 0
all_names = []
names_used = {x:[] for x in bbox_types}
for names in a[0]:
	#print(names)
	for i in os.listdir(dop_dir+'/'+names+'/'):
		#print(dop_dir+'/'+names)

				#print(i[-3:])
		#print('here');l,

		all_names.append(names)
		try:
			ccc+=1
			if zoom in i[-3:]:
				#print(gt_path+'/'+names+'/'+i)
				#exit()
				#gt_zoom_path = gt_path+'/'+names+'/'+i

				#print(i)		

				res_dir = dop_dir+'/'+names+'/'+i+'/Result.mat'
				gt1_mask = scipy.io.loadmat(res_dir)['RegProps_img']['GT1Mask'][0,0][:5]
				#print(gt1_mask)
				#exit()
				#### STRUCTURE - gt1_mask[x,0][-1] will give bounding box corner and length breadth

				gt2_mask = scipy.io.loadmat(res_dir)['RegProps_img'][:,:]['GT2Mask'][0,0][:5]
				
				gt1_region = scipy.io.loadmat(res_dir)['RegProps_img'][:,:]['GT1'][0,0][:5] 
				gt2_region = scipy.io.loadmat(res_dir)['RegProps_img'][:,:]['GT2'][0,0][:5]

				#print(gt1_mask)
				#exit()
				

				img_type_coords = {1:gt1_region,2:gt2_region,3:gt1_mask,4:gt2_mask}
				
				mask = np.zeros_like(plt.imread(orig_dir+'/'+names+'.jpeg'))
				
				all_coords_types  = []

				for item in bbox_types:
					#print()

					if(not os.path.isdir(gt_path+'/'+item)):
						os.makedirs(gt_path+'/'+item)

					gt_zoom_path=gt_path+'/'+item

					color_entry = list(a[a[0]==names][item+c].items())[0][1]
					#print(color_entry)
					#print(item)
					#color_entry = [j.lower() for j in list(color_entry) if j.isalpha() ]
					#print(color_entry)
					#print(list(a[a[0]==names][item+c].items())[0][1])
					#print(names)
					if(not pd.isnull(color_entry ) and not isinstance(color_entry,int) and [j.lower()  for j in color_entry if j.isalpha()][0] in c_to_num):#and not pd.isnull(a[a[0]==names][item+t][0])):
						#print(color_entry)
						color_entry = [j.lower()  for j in color_entry if j.isalpha()]
						#print(color_entry)
						

						type_name = list(a[a[0]==names][item+t].items())[0][1]
						


						gt_type_path = gt_zoom_path+'/'+names+'_'+i+'.png'

						
					#print(color_name)
						
						coords_all = []
						for cc in color_entry:	
							#continue
							c_idx = c_to_num[cc]
							coords = img_type_coords[type_name][c_idx,0][-1][0]
							#print(coords)
							### THE ORDERING OF COOORDS IS left top - right top - right bottom - left bottom
						
							coords_all.append([ [coords[1],coords[0]],[coords[1],coords[0]+coords[2]],[coords[1]+coords[3],coords[0]+coords[2]],[coords[1]+coords[3],coords[0]] ])


							#print(coords)
													
							#color_name = cc
							
								# p1 = [coords[0],coords[1]]
								# p2 = [coords[0]+coords[2],coords[1]]
								# p3 = [coords[0]+coords[2],coords[1]+coords[3]]
								# p4 = [coords[0],coords[1]+coords[3]]

						coords_all = np.array(coords_all)
						#print(coords_all)
						#print(type_name)
						
						if(coords_all.shape[0]>1):
							#print(color_entry)
							#print(coords_all.shape)

							coords_all2 = coords_all.reshape(coords_all.shape[0]*coords_all.shape[1],-1)

							row_coords_min = min(coords_all2[:,0])
							col_coords_min = min(coords_all2[:,1])

							row_coords_max = max(coords_all2[:,0])
							col_coords_max = max(coords_all2[:,1])

							b_box_coord = [[row_coords_min,col_coords_min],[row_coords_min,col_coords_max],[row_coords_max,col_coords_max],[row_coords_max,col_coords_min]]

							#b_box_coord = [min_top_left,max_top_right,max_bot_right,[max_bot_right[0],min_top_left[1]]]
							b_box_coord= np.array(b_box_coord,dtype=np.uint16)

							#print(b_box_coord)

						else:
							b_box_coord= np.array(coords_all,dtype=np.uint16).squeeze()

						### Obtained bounding box, now make segmentation mask using orig_dir
						#print(b_box_coord)
						orig_img = orig_dir+'/'+names+'.jpeg'
						mask_type = np.zeros_like(mask)
						#print(b_box_coord)
						mask_type[int(b_box_coord[0,0]):int(b_box_coord[2,0]),int(b_box_coord[0,1]):int(b_box_coord[1,1])] = 255

						scipy.misc.imsave(gt_type_path,mask_type)

						np.savetxt(gt_zoom_path+'/'+names+'_'+i+'.npy',b_box_coord)

						all_coords_types.append(b_box_coord)

						names_used[item].append(names)
						#print(all_coords_types)
						
						#print(orig_img)

				# if ( ccc > 500 and len(all_coords_types) ==0):
				# 	print(len(all_coords_types))
				# 	print(b_box_coord)
				#all_coords_types=[]
				#print(all_coords_types)
				#count = 66
				fig,ax = plt.subplots(1)
				im = np.array(Image.open(orig_dir+'/'+names+'.jpeg'),dtype=np.uint8)
				ax.imshow(im)

				for coord_type in all_coords_types:
					corner = (coord_type[0,1],coord_type[0,0])
					b = float(coord_type[3,0])-float(coord_type[0,0])
					l = float(coord_type[2,1])-float(coord_type[3,1])
					#print(l,b)

					
					
					rect = patches.Rectangle(corner,l,b,linewidth=2,edgecolor='r',facecolor='none')
					#count+=1
					ax.add_patch(rect)
				#plt.plot()
				#plt.show()

				plt.savefig(gt_path+'/combined/'+names+'_'+i+'.png')
				#fig.close()
				plt.close()

		except IndexError:
			with open('./error_names.txt','a') as f:
				f.write(names+'\n')

			# print(a[a[0]==names]['W'+t][0])
			

			#print(result.shape)

			#### COLOR ORDER = BLUE RED GREEN YELLOW MAGENTA



			#print(len(result['RegProps_img']))
			#print(result['RegProps_img'].shape)
			#break
	#break
			#print(res_dir)

#f.close()

import pickle

pickle.dump(names_used,open('./used_names_types.pkl','wb'))

pickle.dump(all_names,open('./all_names.pkl','wb'))

not_used = {x:[] for x in names_used}

for i in all_names:
	for j in names_used:
		if i not in names_used[j] and i not in not_used[j]:
			not_used[j].append(i)



pickle.dump(not_used,open('./not_used_types.pkl','wb'))





# b = np.array(a)
# print(a.keys())
# print((b.shape))

# # for i in mat['RegProps_img']:
# # 	print(i[0][0].shape)
# # 	print(i[0][0])

# # plt.imshow(plt.imread('/data/DATASETS/WORKABLE/Dicom_Samples/Set2/Gurpreet_Experiments/DATE03282018/12530.60502_0M7GBKVH_I47/1_2_Doppler_03282018_1.0/GT_2_Regions.bmp')),plt.show()

# for names in a[0]:
# 	print(names)

# #for i in mat:
# #	print(i)
# #print(i)



