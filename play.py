import matplotlib.pyplot as plt
import numpy as np
import os
#import pickle
import random
import cv2
from sklearn.cluster import KMeans
from skimage.filters import try_all_threshold
from skimage.filters import threshold_otsu,threshold_local

#list_img = [(cv2.imread('/data/Gurpreet/Doppler_Sample_Images/'+i),i) for i in os.listdir('/data/Gurpreet/Doppler_Sample_Images/') if '.jpeg' in i ]

#img = plt.imread('image7.png')

#print(img.shape)
from scipy import stats
from sklearn.utils import shuffle

train_set = []

def make_train(img_dir):
	### Using 30 images should be enough
	img_dirs = [i for i in os.listdir(img_dir) if '.jpeg' in i][:30]
	print(img_dirs)
	random.shuffle(img_dirs)
	count = 0
	
	img = cv2.imread(img_dir+'/'+img_dirs[0])
			
	train_set = np.zeros((img.shape[0]*img.shape[1],3))
			
	for imgs_name in img_dirs:
		img = cv2.imread(img_dir+'/'+imgs_name)
		imgG = img[:,:,1]
		img_gray = 0.2989*img[:,:,2] + 0.5870*img[:,:,1] + 0.1140*img[:,:,0]
		img_gray = cv2.GaussianBlur(img_gray,(7,7),0)
		img_gray = cv2.dilate(img_gray,None,2)
		img_gray = cv2.erode(img_gray,None,2)

		

		if(count==0):
			train_set = np.zeros((img.shape[0]*img.shape[1],3))
			train = img.reshape(img.shape[0]*img.shape[1],3)[:img.shape[0]*img.shape[1]//3,:]
			shuffle(train)
			train_set = train
			count+=1
		else:
			count+=1
			train = img.reshape(img.shape[0]*img.shape[1],3)[:img.shape[0]*img.shape[1]//3,:]
			shuffle(train)
			
			train_set = np.concatenate((train_set,train),axis = 0)		
		#print(train_set.shape)
	return train_set

import pickle

def train_it(train_img):

	kmeans = KMeans(n_clusters = 10,random_state=0)

	kmeans.fit(train_img)

	pickle.dump(kmeans,open('./model2.pkl','wb'))


#train_img = make_train('/data/DATASETS/WORKABLE/Dicom_Samples/Set2/70_Images/')

#print(train_img.shape)

#shuffle(train_img)


#train_it(train_img=train_img)



model = pickle.load(open('./model.pkl','rb'))

#list_img = [i for i in os.listdir('/data/Gurpreet/Doppler_Sample_Images/') if '.jpeg' in i and '0M7CV8VT.11.jpeg' in i or '0M7AQA0P.19.jpeg' in i]

list_img = [i for i in os.listdir('/data/DATASETS/WORKABLE/Dicom_Samples/Set2/70_Images/') if '.jpeg' in i ]
#list_img = ['/data/DATASETS/WORKABLE/Dicom_Samples/Set2/Gurpreet_Experiments/4750.18091_0KXU5310_I64/1_2_Doppler_21032018/Oimg.bmp']


import scipy.fftpack

from collections import Counter
from scipy.misc import imsave
count=0
import time

#path_of_img = '/data/DATASETS/WORKABLE/Dicom_Samples/Set2/70_Images/'
path_of_img = '/data/DATASETS/WORKABLE/Dicom_Samples/Set2/Cluster_1/'
list_img = [i for i in os.listdir(path_of_img) if '.jpeg' in i]
#img = cv2.imread(path_of_img+'/'+list_img[0])
#labels = model.predict(img.reshape(img.shape[0]*img.shape[1],3))

#labels = labels.reshape(img.shape[0],img.shape[1])


# for i in set(labels):
# 	labels2 = np.zeros_like(img[:,:,0].reshape(-1))
# 	labels2[np.where(labels==i)] = 255
# 	plt.imshow(labels2.reshape((img.shape[0],img.shape[1]))),plt.show()



t11 = time.time()
for name_img in list_img:
	#print(name_img)
	#name_img = '0KXV5LXD.52.jpeg'
	print(name_img)
	img = cv2.imread(path_of_img+name_img)
	#img = cv2.imread(name_img)
	#print(path_of_img+name_img)
	plt.imshow(img),plt.show()
	img_real_gray = 0.2989*img[:,:,2] + 0.597*img[:,:,1] + 0.114*img[:,:,0]
	imgG = img[:,:,1]


	labels = model.predict(img.reshape(img.shape[0]*img.shape[1],3))

	#print(Counter(labels)) class 0 = the class containing the EKG and the brightest portions..

	labels2 = np.zeros_like(labels)

	labels2[np.where(labels==2)] = 1.0

	labels2 = labels2.reshape(img.shape[0],img.shape[1])
	plt.imshow(labels2),plt.show()
	img_filt_r = np.zeros_like(img)

	img_filt_r[:,:,0] = labels2*img[:,:,0] #+ labels2*img[:,:,1] + labels2*img[:,:,2] 

	img_filt_r[:,:,1] = labels2*img[:,:,1] #+ labels2*img[:,:,1] + labels2*img[:,:,2] 

	img_filt_r[:,:,2] = labels2*img[:,:,2] #+ labels2*img[:,:,1] + labels2*img[:,:,2] 



	plt.imshow(img_filt_r),plt.show()

	img_gray = 0.2989*img_filt_r[:,:,2] + 0.5870*img_filt_r[:,:,1] + 0.1140*img_filt_r[:,:,0]
	#img_gray = cv2.GaussianBlur(img_gray,(7,7),0)
	#img_gray = cv2.dilate(img_gray,np.ones((4,8),np.uint8),1)
	#img_gray = cv2.erode(img_gray,None,1)

	#img_gray = cv2.dilate(img_gray,None,1)

	plt.imshow(img_gray),plt.show()	
	img_gray_new = img_gray
	img_gray_new[img_gray>145] = 0

	#cv2.dilate(img_gray_new,np.array([1,1]),2)
	#img_gray_new2 = cv2.morphologyEx(img_gray_new,cv2.MORPH_CLOSE,np.ones((1,20)))
	img_gray_new2 = cv2.dilate(img_gray_new.copy(),np.ones((4,30)),2)
	#img_gray_new2 = cv2.erode(img_gray_new.copy(),np.ones((1,30)),2)
	#img_gray_new2 = cv2.GaussianBlur(img_gray_new2,(1,11),10)
	#img_gray_new2 = cv2.dilate(img_gray_new2,np.ones((1,20)),1)
	
	#img_gray_new2 = cv2.morphologyEx(img_gray_new,cv2.MORPH_CLOSE,np.ones((3,20)))
	#img_gray_new2 = cv2.dilate(img_gray_new2,np.ones((1,20)),1)
	
	
	thresh = threshold_otsu(img_gray_new2)
	thresh = np.array(thresh,dtype=np.uint8)
	
	gray_im = img_gray_new2>thresh
	gray_im = np.array(gray_im,dtype=np.uint8)
	
	plt.imshow(gray_im),plt.show()

	output = cv2.connectedComponentsWithStats(gray_im,connectivity=4)


	sizes = output[2][:,-1]
	max_s = 1
	# for i in range(2,output[0]):
	# 	if(sizes[i]>max_s):
	# 		max_l = i
	# 		max_s = sizes[i]
	init = 0
	img_seg = np.zeros_like(gray_im)

	length_list = []
	index_list = []



	for i in range(2,output[0]):

		img22= np.zeros_like(gray_im)
		img22[output[1]==i]=1.0
		coord_loc = np.where(img22>0)
		
		leng = coord_loc[1].max()-coord_loc[1].min()
		breadth = coord_loc[0].max()-float(coord_loc[0].min())

		breadth_avg = coord_loc[0].sum()/float(len(coord_loc[0]))

		#print('ab')
		#if(leng>init):
			
		#print(coord_loc[0])
		#print('aa')
		#print(coord_loc[1])
		
		#breadth_list = coord_loc[:,]


		length_list.append(leng)
		index_list.append(i)

		if(leng>init and abs(breadth + coord_loc[0].min() - breadth_avg) < img22.shape[0]/3  and len(coord_loc[0]) < img_seg.shape[0]*img_seg.shape[1]//20):
			init = leng
			max_l = i
			img_seg = img22
			breadth_best = coord_loc[0].max()-float(coord_loc[0].min())

			best_area = len(coord_loc[0])
			#print(len(coord_loc[0]))

			breadth_avg_best = coord_loc[0].sum()/float(len(coord_loc[0]))

	# print(best_area)
	# print(breadth_best)
	# print(breadth_avg_best)
	#print(breadth + coord_loc[0].min())
	#print(breadth_avg)
	#if(init < img_seg.shape[1]/3)



	plt.subplot(211)
	plt.imshow(img)
	plt.subplot(212)
	plt.imshow(img_seg)
	
	ecg_part_temp = np.zeros_like(img_gray.shape)
	
	#ecg_part_temp = np.zeros_like(img)

	plt.imshow(img_seg),plt.show()
	#print(img.shape)
	ecg_part_temp = img[:,:,0]*img_seg
	
	#ecg_part_temp[:,:,0] = img[:,:,0]*img_seg
	#ecg_part_temp[:,:,1] = img[:,:,1]*img_seg
	#ecg_part_temp[:,:,2] = img[:,:,2]*img_seg

	plt.imshow(ecg_part_temp),plt.show()
	#continue
	#print(threshold_otsu(ecg_part_temp))
	plt.imshow(ecg_part_temp),plt.show()
	thresh2 = ecg_part_temp > threshold_otsu(ecg_part_temp)
	
	#thresh2 = cv2.dilate(thresh2.astype(np.uint8),np.ones((1,5)),1)
	#thresh2 = cv2.erode(thresh2.astype(np.uint8),np.ones((1,5)),1)
	#plt.subplot(211)
	# plt.imshow(thresh2)
	# plt.show()
	#exit()
	non_zero_coord = np.array(np.where(thresh2>0)).T

	sorted_coords = np.array(sorted(list(non_zero_coord),key = lambda x:x[1]))

	# print(sorted_coords)
	min_point = sorted_coords[:,0].max()	
	avg_point = sorted_coords[:,0].mean()
	
	relative_values = - sorted_coords[:,0] + float(min_point)
	relative_values = - sorted_coords[:,0] + float(avg_point)

	#print(relative_values)
	#plt.plot(relative_values),plt.show()
	#plt.plot(relative_values),plt.show()
	N = len(relative_values)

	x = np.linspace(sorted_coords[:,1].min(),sorted_coords[:,1].max()+1,N)
	w = scipy.fftpack.rfft(relative_values)
	
	#plt.plot(w),plt.show()
	#exit()
	f = scipy.fftpack.rfftfreq(N,x[1]-x[0])
	#plt.plot(f),plt.show()

	spectrum = w**2
	#cutoff_idx = spectrum[
	
	w2 = w.copy()
	#print(w2.shape)
	f_cut = len(w2)//20
	#print(f_cut)
	w2[f_cut:]=0

	y2 = scipy.fftpack.irfft(w2)

	plt.subplot(211)
	plt.plot(relative_values)
	plt.subplot(212)
	plt.plot(y2)#,plt.show()

	plt.savefig('./ECG_part/fiter_signal_new_'+str(name_img)+'.png')
	
	plt.close()
	thresh2[thresh2>0] = 255
	
	#plt.imshow(thresh2),plt.show()
	#plt.plot(y2),plt.show()
	#plt.close()
	imsave('./ECG_part/fiter_signal_new_'+str(name_img)+'_ECG_SEG.png',thresh2)
	np.savetxt('./ECG_part/fiter_signal_new_'+str(name_img)+'_coords.npy',sorted_coords)
	
	

	#plt.plot(relative_values),plt.show()

	# mng = plt.get_current_fig_manager()
	# mng.window.showMaximized()
	# plt.show()
	# plt.close()
	
	count+=1
	
#print((time.time()-t11)/10)	
#'''