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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
from tqdm import tqdm
from hw2_evaluation_task import eval
from models import *

__all__ = ['vgg16_bn']
model_urls = {
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
}
ROOT =os.getcwd()
TRAIN_DIR = 'hw2_train_val/train15000'
VAL_DIR = 'hw2_train_val/val1500'
train_image_dir = join(ROOT, TRAIN_DIR, 'images')
train_label_dir = join(ROOT, TRAIN_DIR, 'labelTxt_hbb')

val_image_dir = join(ROOT, VAL_DIR, 'images')
val_label_dir = join(ROOT, VAL_DIR, 'labelTxt_hbb')
EPOCH = 80
BATCH_SIZE = 48
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def main():
    

    model = Yolov1_vgg16bn(pretrained=True).to(device)
   # model.load_state_dict(torch.load('yolo_model_v1.pth'))
    data_loader = BboxLabelDataset(train_image_dir, train_label_dir)
    data_loader = DataLoader(data_loader, batch_size=BATCH_SIZE, shuffle=True)
    val_data_loader = BboxLabelDataset(val_image_dir, val_label_dir)
    val_data_loader = DataLoader(val_data_loader, batch_size=BATCH_SIZE, shuffle=True)
    loss_func = yoloLoss().to(device)
    for ep in range(EPOCH):
        print('Epoch : ', ep)
        
        if ep < 10:
            LR = 1e-4
        elif ep < 30:
            LR = 1e-4
        elif ep < 60:
            LR = 5e-5
        else:
            LR = 1e-5

        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
        total_loss = 0

        for step, batch in enumerate(tqdm(data_loader, ncols=50)):
            
            optimizer.zero_grad()    
            
            image_batch = Variable(batch['image']).to(device)
            label_batch = Variable(batch['label']).to(device)

            
            
            output = model(image_batch)
            loss = loss_func(output, label_batch)
            total_loss += loss.data
            loss.backward()                 
            optimizer.step()
            

        print(total_loss / len(data_loader))

        for step, batch in enumerate(tqdm(val_data_loader, ncols=50)):
            
            image_batch = Variable(batch['image']).to(device)
            label_batch = Variable(batch['label']).to(device)
            filename = batch['filename']

            output = model(image_batch)
            
            for index, data in enumerate(output):
                data = data.detach().cpu().numpy()
                row, col = np.where(data[..., 4] > 0.5)
            
                with open(join(VAL_DIR, 'pred', filename[index] + '.txt'), 'w+') as f:
                    line = []
                    # print(join(predict_dir, filename + '.txt'))

                    for r, c in zip(row, col):

                        p = data[r, c]
                        i, j = r, c
                        x, y, w, h = p[:4]
                        x = ((j+x)*64)*512//448
                        y = ((i+y)*64)*512//448
                        w = int(w*512)
                        h = int(h*512)
                        
                        argmax = np.argmax(p[10:])

                        classname = class_table[argmax]
                        class_specific = p[4] * p[10+argmax]
                        # print(classname)
                        # rect = patches.Rectangle((int(x-w/2),int(y-h/2)),w,h,linewidth=1,edgecolor='r',facecolor='none')
                        # ax[index].add_patch(rect)

                        line = []
                        line += [round(x - w/2, 1), round(y - h/2 ,1)]
                        line += [round(x + w/2, 1), round(y - h/2 ,1)]
                        line += [round(x + w/2, 1), round(y + h/2 ,1)]
                        line += [round(x - w/2, 1), round(y + h/2 ,1)]
                        line += [classname, class_specific]
                        for word in line:
                            f.write(str(word)+' ')
                        f.write('\n')

                    f.close()
                
            
        eval(VAL_DIR+'/pred/', VAL_DIR + '/labelTxt_hbb/')

        if ep % 10 == 0:
            torch.save(model.state_dict(), './yolo_'+str(ep)+'_model.pth')
        torch.save(model.state_dict(), 'yolo_model_v1.pth')


    



if __name__ == '__main__':
    
    main()
