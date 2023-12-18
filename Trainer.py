

# from itertools import Predicate
import torch
import torch.nn as nn
from deap_model.mmformer import Model
import torch.nn.functional as F
import sys
# sys.path.append('E:\\Multimodal-Dense-GCN-5')
from utils import torch_utils


class mmformer_network(nn.Module):
    def __init__(self, opt):
        super(mmformer_network,self).__init__()
        self.opt = opt
        cls_num = opt['num_class']
        self.trainer = Model(opt)
        self.fc = nn.Linear(cls_num*16*16, cls_num)
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.parameters = [p for p in self.trainer.parameters() if p.requires_grad]
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'], opt['weight_decay'])


    def train (self, modal1, modal2, train_y):
       
        self.trainer.train()
        self.optimizer.zero_grad()
        
        las_targ, sep_targs, sep_l2, fuse_targs = self.trainer(modal1, modal2, is_training=True)
        ######Dsep_loss
        Sep_loss = 0
        Sep_l2_loss = 0
        for i in range(len(sep_targs)):
            sep_out = self.fc(sep_targs[i].reshape(sep_targs[i].shape[0], -1))
            sep_loss = self.criterion(sep_out, train_y.squeeze(1).long())
            Sep_loss += sep_loss
            l2_loss = sep_l2[i] * sep_l2[i]
            Sep_l2_loss += l2_loss

        ######Fuse_loss: Layer_sum
        Fuse_loss=0
        for i in range(len(fuse_targs)):
            fuse_out = self.fc(fuse_targs[i].reshape(fuse_targs[i].shape[0], -1))
            fuse_loss = self.criterion(fuse_out, train_y.squeeze(1).long())
            Fuse_loss += fuse_loss

        ##########全局loss
        las_out = self.fc(las_targ.reshape(las_targ.shape[0], -1))
        Las_loss = self.criterion(las_out, train_y.squeeze(1).long())
        log = F.softmax(las_out, 1)

        # loss = self.opt['lamda1'] *Fuse_loss + Sep_loss + (self.opt['weight'] * Sep_l2_loss) + self.opt['lamda2'] *Las_loss

        # loss = Sep_loss + self.opt['weight'] * Sep_l2_loss   ###Loss1
        # loss = Fuse_loss   ###Loss2
        # loss = Las_loss   ###Loss3
        # loss = Fuse_loss + Sep_loss + (self.opt['weight'] * Sep_l2_loss)   ###loss1+loos2
        loss = Fuse_loss + Las_loss   ###loss1+loss3
        # loss = Sep_loss + (self.opt['weight'] * Sep_l2_loss) + Las_loss    ###loss2+loss3
        
        #####backward
        loss.backward()
        self.optimizer.step() 
        self.optimizer.zero_grad()

        return log, loss.item()


    def predict(self,modal1, modal2, test_y):
        self.trainer.eval()
        # for layer in self.trainer:
        #     pred = layer(test_x)
        pred = self.trainer(modal1, modal2, is_training=False)
        las_pred = self.fc(pred.reshape(pred.shape[0], -1))
        loss = self.criterion(las_pred, test_y.squeeze(1).long())
        probs = F.softmax(las_pred, 1)
        # preds = np.argmax(probs.data.cpu().numpy(), axis=1)
        return probs, loss


   