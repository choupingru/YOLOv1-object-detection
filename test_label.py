import torch.nn as nn
import math
import torch
import torch.utils.model_zoo as model_zoo
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from yolo_loss import yoloLoss
from data_preprocessing import BboxLabelDataset
from os.path import join
import os
import nms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms 
from PIL import Image

ROOT = os.getcwd()
TRAIN_DIR = 'hw2_train_val/train15000'
train_image_dir = join(ROOT, TRAIN_DIR, 'images')
train_label_dir = join(ROOT, TRAIN_DIR, 'labelTxt_hbb')

data_loader = BboxLabelDataset(train_image_dir, train_label_dir)
data_loader = DataLoader(data_loader, batch_size=2, shuffle=False)

for step, batch in enumerate(data_loader):

	image_batch = batch['image']
	label_batch = batch['label']

	for image, label in zip(image_batch, label_batch):
		image = np.array(image)
		image = image.transpose((1,2,0))
		
		fig, ax = plt.subplots(1)
		ax.imshow(image)
		for i in range(7):
			for j in range(7):
				x, y, w, h = label[i][j][:4]
				x = (j+x)*64
				y = (i+y)*64
				w *= 448
				h *= 448
				if label[i][j][4] == 1:
					rect = patches.Rectangle((int(x-w/2),int(y-h/2)),w,h,linewidth=1,edgecolor='r',facecolor='none')
					ax.add_patch(rect)

		plt.show()
