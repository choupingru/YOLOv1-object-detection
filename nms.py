import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def NMS_IOU(x, y):
    # x: [dx, dy, w, h]
    # y: [dx, dy, w, h]
	x = x.float()
	y = y.float()

	box1_lt = torch.cat((x[0:1]*(448//7) - x[2:3]*448, x[1:2]*(448//7) - x[3:4]*448), dim=0).to(device)
	box1_rb = torch.cat((x[0:1]*(448//7) + x[2:3]*448, x[1:2]*(448//7) + x[3:4]*448), dim=0).to(device)

	box2_lt = torch.cat((y[0:1]*(448//7) - y[2:3]*448, y[1:2]*(448//7) - y[3:4]*448), dim=0).to(device)
	box2_rb = torch.cat((y[0:1]*(448//7) + y[2:3]*448, y[1:2]*(448//7) + y[3:4]*448), dim=0).to(device)

	xA = torch.max(box1_lt[0:1], box2_lt[0:1]).to(device)
	yA = torch.max(box1_lt[1:2], box2_lt[1:2]).to(device)
	xB = torch.min(box1_rb[0:1], box2_rb[0:1]).to(device)
	yB = torch.min(box1_rb[1:2], box2_rb[1:2]).to(device)

	interArea = torch.max(torch.zeros(xB.size()).to(device), xB - xA + 1) * torch.max(torch.zeros(xB.size()).to(device), yB - yA + 1)
	box1Area = (box1_rb[0:1] - box1_lt[0:1] + 1) * (box1_rb[1:2] - box1_lt[1:2] + 1)
	box2Area = (box2_rb[0:1] - box2_lt[0:1] + 1) * (box2_rb[1:2] - box2_lt[1:2] + 1)

	iou = (interArea)/(box1Area + box2Area - interArea)
    
	return iou

def NMS(output):
	'''
		output : (BATCH_SIZE, 7, 7, 26)
	'''
	res = []
	for batch in output:
		candidate = torch.Tensor(torch.zeros(23)).to(device)
		candidate = candidate.view(1, -1)
		for i in range(7):
			for j in range(7):
				class_prob = batch[i][j][10:].to(device)
				predict_class_prob = F.softmax(class_prob).to(device)
				box1 = torch.cat((batch[i][j][:5].to(device), predict_class_prob)).to(device)
				box1 = torch.cat((box1, torch.Tensor([i, j]).to(device))).to(device)
				
				box2 = torch.cat((batch[i][j][5:10].to(device), predict_class_prob)).to(device)
				box2 = torch.cat((box2, torch.Tensor([i, j]).to(device))).to(device)

				if box1[4] >= 0.1:
					
					box1 = box1.view(1, -1)
					candidate = torch.cat((candidate, box1), dim=0)
				if box2[4] >= 0.1:
					box2 = box2.view(1, -1)
					candidate = torch.cat((candidate, box2), dim=0)
		candidate = candidate[1:]
		
		this_batch = []
		while len(candidate) != 0:
			argmax = torch.argmax(candidate[:,4:5], dim=0).to(device)
			max_box = candidate[argmax][0].to(device)
			this_batch.append(max_box)	
			
			c = 0
			for index, score in enumerate(candidate):
				
				iou = NMS_IOU(max_box, score)
				if iou >= 0.5:
					candidate = torch.cat((candidate[:index-c].to(device), candidate[index+1-c:].to(device)))
					c+=1
		res.append(this_batch)
		
	return res
		