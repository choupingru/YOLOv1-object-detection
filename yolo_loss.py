import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class yoloLoss(nn.Module):

    def __init__(self):
        super(yoloLoss, self).__init__()
        self.S = 7
        self.B = 2
        self.num_class = 16
        self.lambdaLoc = 10
        self.lambdaNoObj = 0.5
    def IOU(self, x, y):
        # x: N*S*S, [dx, dy, w, h]
        # y: N*S*S, [dx, dy, w, h]
        x = x.float().to(device)
        y = y.float().to(device)
        box1_lt = torch.cat((x[:,0:1]*64 - x[:,2:3]*224, x[:,1:2]*64 - x[:,3:4]*224), dim=1).to(device)
        box1_rb = torch.cat((x[:,0:1]*64 + x[:,2:3]*224, x[:,1:2]*64 + x[:,3:4]*224), dim=1).to(device)

        box2_lt = torch.cat((y[:,0:1]*64 - y[:,2:3]*224, y[:,1:2]*64 - y[:,3:4]*224), dim=1).to(device)
        box2_rb = torch.cat((y[:,0:1]*64 + y[:,2:3]*224, y[:,1:2]*64 + y[:,3:4]*224), dim=1).to(device)

        xA = torch.max(box1_lt[:,0:1], box2_lt[:,0:1]).to(device)
        yA = torch.max(box1_lt[:,1:2], box2_lt[:,1:2]).to(device)
        xB = torch.min(box1_rb[:,0:1], box2_rb[:,0:1]).to(device)
        yB = torch.min(box1_rb[:,1:2], box2_rb[:,1:2]).to(device)

        interArea = torch.max(torch.zeros(xB.size()).to(device), xB - xA + 1) * torch.max(torch.zeros(yB.size()).to(device), yB - yA + 1)
        
        box1Area = (box1_rb[:,0:1] - box1_lt[:,0:1] + 1) * (box1_rb[:,1:2] - box1_lt[:,1:2] + 1)
        box2Area = (box2_rb[:,0:1] - box2_lt[:,0:1] + 1) * (box2_rb[:,1:2] - box2_lt[:,1:2] + 1)
        
        iou = (interArea)/(box1Area + box2Area - interArea)
        
        return iou

    def forward(self, predict, target):
        
        '''
            Arg:
                predict : [N, S, S, (5 * B) + num_class ]
                target  : [N, S, S, (5 * B) + num_class ]
        '''
        N = target.size(0)
        predict = predict.view(N, -1, self.B * 5 + self.num_class ).to(device)
        target = target.view(N, -1, self.B * 5 + self.num_class ).to(device)
        predict = predict.view(-1, self.B * 5 + self.num_class).to(device)
        target = target.view(-1, self.B * 5 + self.num_class).to(device)
        '''
        predict , target : [N * 49, (5 * B) + num_class]
        '''

        obj_mask = target[:,4:5].to(device)
        no_obj_mask = torch.abs(obj_mask - 1).to(device)
        
        obj_index = obj_mask.nonzero()[:,0].to(device)
        no_obj_index = no_obj_mask.nonzero()[:,0].to(device)

        '''
        obj_pred, obj_tar : [N , 26]
        '''
        obj_predict = predict[obj_index].to(device)
        obj_target = target[obj_index].to(device)

        obj_pred_box1 = obj_predict[:, :5].to(device)
        obj_pred_box2 = obj_predict[:, 5:10].to(device)
        
        box1_iou = self.IOU(obj_pred_box1, obj_target).to(device)
        box2_iou = self.IOU(obj_pred_box2, obj_target).to(device)
        iou_combine = torch.cat((box1_iou, box2_iou), dim=1).to(device)
        
        iou_argmax = torch.argmax(iou_combine, dim=1).to(device)
        iou_argmax = iou_argmax.view(-1, 1)
        iou_max = torch.gather(iou_combine, dim=1, index=iou_argmax).to(device)
        
        obj_target[:, 4:5] = iou_max
        obj_target[:, 9:10] = iou_max

        iou_argmax = torch.cat((iou_argmax*5, iou_argmax*5+1, iou_argmax*5+2, iou_argmax*5+3, iou_argmax*5+4), dim=1).to(device)

        valid_pred_bbox = torch.gather(obj_predict, dim=1, index= iou_argmax).to(device)
        valid_tar_bbox = torch.gather(obj_target, dim=1, index=iou_argmax).to(device)
        
        xy_loss = F.mse_loss(valid_pred_bbox[:,:2].double(), valid_tar_bbox[:,:2].double(), reduction='sum').to(device)
        wh_loss = F.mse_loss(torch.pow(valid_pred_bbox[:,2:4].double(),0.5), torch.pow(valid_tar_bbox[:,2:4].double(), 0.5), reduction='sum').to(device)

        obj_confidence_loss = F.mse_loss(valid_pred_bbox[:,4:5].double(), valid_tar_bbox[:,4:5].double(), reduction='sum').to(device)

        '''
        no_obj_pred, no_obj_tar : [N, 26]
        '''
        no_obj_predict = predict[no_obj_index].to(device)
        no_obj_target = target[no_obj_index].to(device)
        no_obj_confidence_loss_box1 = F.mse_loss(no_obj_predict[:,4:5].double(), no_obj_target[:,4:5].double(), reduction='sum').to(device)
        no_obj_confidence_loss_box2 = F.mse_loss(no_obj_predict[:,9:10].double(), no_obj_target[:,9:10].double(), reduction='sum').to(device)
      
        #pred_class = F.softmax(obj_predict[:,10:]).to(device)
        #target_class = F.softmax(obj_target[:, 10:])
        class_loss = F.mse_loss(obj_predict[:,10:].double(), obj_target[:, 10:].double(), reduction='sum').to(device)

        total_loss = self.lambdaLoc * xy_loss + self.lambdaLoc * wh_loss +  obj_confidence_loss + self.lambdaNoObj * (no_obj_confidence_loss_box1+no_obj_confidence_loss_box2) +  2 * class_loss
        
        return total_loss 

