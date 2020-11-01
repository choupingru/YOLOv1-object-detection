import numpy as np
import os
from os.path import join
from os import listdir
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.autograd import Variable
from torchvision import transforms 

ROOT = os.getcwd()
DIR = 'hw2_train_val/train15000'

transform_train = transforms.Compose([
	transforms.Resize(448),
    transforms.ToTensor()
])

class_table = {
	'plane':0, 'ship':1,'storage-tank':2,'baseball-diamond':3,'tennis-court':4,'basketball-court':5,
	'ground-track-field':6,'harbor':7,'bridge':8,'small-vehicle':9,'large-vehicle':10,'helicopter':11,
	'roundabout':12,'soccer-ball-field':13,'swimming-pool':14,'container-crane':15
}



class BboxLabelDataset(Dataset):

	def __init__(self, img_dir, label_dir, transform=transform_train):

		self.img_dir = img_dir
		self.label_dir = label_dir
		self.transform = transform
		self.files = sorted(listdir(self.img_dir))

	def __len__(self):
		return len(self.files)

	def __getitem__(self, idx):

		# 0000.jpg -> 0000
		filename = self.files[idx].split('.')[0]

		# # read & resize image
		image = Image.open(join(self.img_dir, filename+'.jpg'))
		image = transform_train(image)

		# # label infomation

		infos = open(join(self.label_dir, filename+'.txt'))
		result = np.zeros(shape=(7,7,26))
		
		for line in infos:
			line = line.replace('\n', '')
			line = line.split(' ')
			
			classname, difficulty = line[-2:]
			points = np.array([float(num)*448/512 for num in line[:8]])
			points = np.reshape(points, (-1, 2))
			
			lt, rt, rb, lb = points

			center_x, center_y = ((lt+rt)/2)[0], ((lt+lb)/2)[1]
			
			height, width = (lb - lt)[1], (rt - lt)[0]
			grid_row, grid_col = int(center_y/64), int(center_x/64)
			x, y = (center_x%64)/64, (center_y%64)/64
			w, h = width / 448, height / 448

			class_ = np.zeros(shape=(16))
			class_[class_table[classname]] = 1

			bbox = np.array([x, y, w, h, 1])
			final = np.hstack((bbox, bbox, class_))
			result[grid_row][grid_col] = final

		sample ={
		'filename':filename,
		'image':image,
		'label':result
		}		

		return sample


# if __name__ == '__main__':
# 	ROOT =os.getcwd()
# 	TRAIN_DIR = 'hw2_train_val/train15000'
# 	train_image_dir = join(ROOT, TRAIN_DIR, 'images')
# 	train_label_dir = join(ROOT, TRAIN_DIR, 'labelTxt_hbb')
# 	loader =BboxLabelDataset(train_image_dir, train_label_dir)
# 	sample = loader[0]
# 	print(sample)




