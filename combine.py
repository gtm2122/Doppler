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

#for i in os.listdir(dop_dir):

#print(a[0])

#print(a['Wcolor'])
import re

t = 'Imgtype'
c = 'color'

bbox_types = ['W','E','V','C','T','S']

c_to_num = {'b':0,'r':1,'g':2,'y':3,'m':4}

# f = open('./error_names.txt','w')
# f.close()

for names in a[0]:
	print(names)
	for i in os.listdir(dop_dir+'/'+names+'/'):
		#print(dop_dir+'/'+names)
		#print(i[-3:])
		#print('here')
		if zoom in i[-3:]:
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

			for item in bbox_types:
				#print()
				color_entry = list(a[a[0]==names][item+c].items())[0][1]
				#print(color_entry)
				print(item)
				#color_entry = [j.lower() for j in list(color_entry) if j.isalpha() ]
				#print(color_entry)
				#print(list(a[a[0]==names][item+c].items())[0][1])
				if(not pd.isnull(color_entry )):#and not pd.isnull(a[a[0]==names][item+t][0])):
					#print(color_entry)
					color_entry = [i.lower()  for i in color_entry if i.isalpha()]
					print(color_entry)
					

					type_name = list(a[a[0]==names][item+t].items())[0][1]
						
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
					print(type_name)
					if(coords_all.shape[0]>1):
						#print('min')
						#print(coords_all)
						#print('mix')
						min_top_left = min(coords_all,key=lambda x:x[0,0])[0]
						max_top_right = [min_top_left[0],max(coords_all,key=lambda x:x[1,1])[1,1]]
						max_bot_right = [max(coords_all,key=lambda x:x[2,0])[2][0],max(coords_all,key=lambda x:x[1,1])[1,1]]

						b_box_coord = [min_top_left,max_top_right,max_bot_right,[max_bot_right[0],min_top_left[1]]]

						#print(b_box_coord)
						#print('here')
						#exit()
						#min_bot_r = -np.inf

						#max_top_
						# print(names)
						# print('b_box_coord = ')

						# print(b_box_coord)
						
					else:

						b_box_coord = coords_all.squeeze()

					### Obtained bounding box, now make segmentation mask using orig_dir

					orig_img = orig_dir+'/'+names+'.jpeg'
					print(orig_img)


					# print(names)
					# print('b_box_coord = ')

					# print(b_box_coord)
					
					#print(coords_all)
					#exit()

					#print(coords_all.shape)

					# except:

					# 	with open('./error_names.txt','a') as f:
					# 		f.write(names+'\n')

					# 	continue

						#print(coords)
						#break
					# else:

					# 	print(re.split(a[a[0]==names][item+c],'[+]'))

						#break

					#print(item+c,pd.isnull(a[a[0]==names][item+c][0])) ### color
					#print(item+t,pd.isnull(a[a[0]==names][item+t][0])) ### type
				#print(a[a[0]==names][item+c][0])

			# print(a[a[0]==names]['W'+t][0])
			

			#print(result.shape)

			#### COLOR ORDER = BLUE RED GREEN YELLOW MAGENTA



			#print(len(result['RegProps_img']))
			#print(result['RegProps_img'].shape)
			#break
	#break
			#print(res_dir)

f.close()


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



