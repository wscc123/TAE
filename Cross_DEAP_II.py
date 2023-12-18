
import time
from datetime import datetime
import torch
import h5py
import numpy as np
import argparse
import torch.utils.data as Data
from utils.dataloader import MyDataSet_train
from utils.dataloader import MyDataSet_test
from Data_Incomplete import From_Incomplete_Data_DEAP
from deap_model.Trainer import mmformer_network
# from deap_model.Trainer import Hybird_network
# from model.Test import *
import os
import scipy.io as io
import torch.nn.parallel
import time
import random
import pandas
import torchvision.transforms as transforms
# from model import adj
from utils import torch_utils
# from torch.utils.tensorboard import SummaryWriter

# for lam in [0.001, 0.01, 0.1, 1, 10, 100]:  ###0.001, 0.01, 0.1, 1, 10, 100
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  
parser = argparse.ArgumentParser()
parser.add_argument('--num_epoch', type=int, default=20, help='Number of total training epochs.')
parser.add_argument('--tr_batch_size', type=int, default=242, help='Training batch size.')
parser.add_argument('--te_batch_size', type=int, default=242, help='Testing batch size.')
parser.add_argument('--num_heads', default=1, type=int, help='The number of Multihead')
parser.add_argument('--num_class', type=int, default= 2, help='Sample of class')
parser.add_argument('--lamda1', default=0.01, type=float, help="Loss weight1")
parser.add_argument('--lamda2', default= 1, type=float, help="Loss weight2")
parser.add_argument('--weight', default=1e-9, type=float, help="Loss_Desp weight")
parser.add_argument('--missrate', default=0.3, type=float, help="Miss rate: 0.1,0.3,0.5,0.7")
parser.add_argument('--lr', type=float, default= 0.00001, help='Applies to sgd and adagrad.')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam', help='Optimizer: sgd, adagrad, adam or adamax.')
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')
parser.add_argument('--cpu',  action='store_true', help='Ignore CUDA.')
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--seed', default=1024, type=int)

# make opt
args = parser.parse_args()
opt = vars(args)

##显卡设置
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

init_time = time.time()

dataset = io.loadmat('/share/chengcheng/hybird/DEAP-dataset/Face/Face_patch_imgdata_nor.mat')
XX1 = dataset["Face_patch_imgdata_nor"]  ####9680*16*3*32*32

dataset = h5py.File('/share/chengcheng/hybird/DEAP-dataset/EEG_all_DE_lap.mat','r')
one_data1 = dataset["de_beta"]    ############选择不同波段
X_e = np.array(one_data1).transpose(2,1,0)    ####9680*32*10
XX2 = X_e[0:XX1.shape[0],:]
YY = dataset["labels_A"]
all_label = np.array(YY)
labs = all_label.transpose(1,0)
YY = labs[0:XX1.shape[0],:]

########增大样本点/中心剪裁-crop
trans_crop1 = transforms.RandomCrop(size=(32, 32)) 
trans_crop2 = transforms.RandomCrop(size=(32, 10))   # SEED： 185*108 ， SEED/session1： 12*108, SEED/session2： 10*108, SEED/session3： 14*108

crop1 = trans_crop1(torch.from_numpy(XX1.reshape(-1, 32, 32))).reshape(XX1.shape[0],XX1.shape[1],XX1.shape[2],XX1.shape[3],XX1.shape[4]) 
crop2 = trans_crop2(torch.from_numpy(XX2))

XX1_large = np.concatenate((XX1, crop1.numpy()), axis= 0 ) 
XX2_large =  np.concatenate((XX2, crop2.numpy()), axis= 0 )
labels = np.concatenate((YY, YY), axis=0)

# ######创建csv文件，保存最后结果
df = pandas.DataFrame(columns=['time', 'Fold','epoch_acc','epoch_acc1','epoch_acc2','Bepoch_acc', 'Bepoch_acc1','Bepoch_acc2']) #列名

def GetNowTime():#获取当前时间并以年月日时间方式显示
    return time.strftime("%m%d%H%M%S",time.localtime(time.time()))

root_path = '/home/chengcheng/Incomplete-Multimodal/Results/DEAP/Cross/Arousal'
# csv_name = 'Incomplete_beta_'  +  str(args.missrate) + '_' + str(args.lamda1) + '_' + str(args.lamda2) + '.csv'
csv_name = 'Loss1_3_beta_'  +  str(args.missrate) + '.csv'
# local_time=time.localtime()[0:5]
# csv_name = 'Incomplete_beta_'  +  str(args.missrate) + '_{:02d}_{:02d}{:02d}_{:02d}{:02d}'\
#                         .format(local_time[0], local_time[1], local_time[2],local_time[3], local_time[4]) + '.csv'

# local_time=time.localtime()[0:3]
# csv_name = 'Incomplete_beta_'  +  str(args.missrate) + '_{:02d}_{:02d}_{:02d}'\
#                         .format(local_time[0], local_time[1], local_time[2]) + '.csv'

# df.to_csv(os.path.join(root_path,csv_name), index=False )

############data transform-normalization
XX1_sor = torch.from_numpy(XX1_large.reshape(XX1_large.shape[0], -1))  ###9680*16*3*32*32
means = torch.mean(XX1_sor, dim=1)
stds = torch.std (XX1_sor, dim=1)
trans_norm = transforms.Normalize(means, stds)
XX1_nor = trans_norm(XX1_sor.reshape(XX1_large.shape[0], XX1_large.shape[1], -1)).reshape(XX1_large.shape[0], 
                        XX1_large.shape[1],XX1_large.shape[2],XX1_large.shape[3], XX1_large.shape[4])
modal1 = XX1_nor.numpy()

XX2_sor = torch.from_numpy(XX2_large.reshape(XX2_large.shape[0], -1))  ###9680*(32*10)
means = torch.mean(XX2_sor, dim=1)
stds = torch.std (XX2_sor, dim=1)
trans_norm = transforms.Normalize(means, stds)
XX2_nor = trans_norm(XX2_sor.reshape(XX2_large.shape[0], -1, XX2_large.shape[2]))
modal2 = XX2_nor.numpy()

###########Incomplete-data
modal_1, modal_2, labels = From_Incomplete_Data_DEAP(args.missrate, modal1, modal2, labels)  ###Face, EEG

#########shuffle data
index = np.array(range(0, len(labels))) 
np.random.shuffle(index)#乱序序号
Modal_2 = modal_2[index]#按乱序序号取数据
Modal_1 = modal_1[index]#按乱序序号取数据
all_labels  = labels[index]#按乱序取标签

#####计算平均结果用
bat =0
bat1 =0
bat2 =0

e =0

e1 =0
e2 =0

fold = 5
print ('#############training############')
global_start_time = time.time()
for cur_fold in range (0, fold):
    print ("cur_fold:", cur_fold +1)
    fold_size = Modal_2.shape[0] // fold 
    indexes_list = [i for i in range(len(Modal_2))]
    split_list = [i for i in range(cur_fold*fold_size,(cur_fold+1)*fold_size)]#分第几折
    split = np.array(split_list)
    Test_x_modal_2 = Modal_2[split] #45
    Test_y = all_labels[split]
    Test_x_modal_1 = Modal_1[split]

    split = np.array(list(set(indexes_list)^set(split_list)))
    Train_x_modal_2 = Modal_2[split] # 180
    Train_y = all_labels[split]
    Train_x_modal_1 = Modal_1[split]

    M1= Train_x_modal_2.shape[0] // opt['tr_batch_size']
    M11= M1 * opt['tr_batch_size']
    Train_x_modal_2 = Train_x_modal_2[0:M11,:,:]
    Train_x_modal_1 = Train_x_modal_1[0:M11,:,:]
    Train_y = Train_y[0:M11,:]

    M2= Test_x_modal_2.shape[0] // opt['te_batch_size']
    M22= M2 * opt['te_batch_size']
    Test_x_modal_2 = Test_x_modal_2[0:M22,:,:]
    Test_x_modal_1 = Test_x_modal_1[0:M22,:,:]
    Test_y = Test_y[0:M22,:]
    
    print("Train samples:", Train_y.shape[0])
    print("Test samples:", Test_y.shape[0])
    print("Train_batch_num:", Train_y.shape[0] // opt['tr_batch_size'])
    print("Test_batch_num:", Test_y.shape[0] // opt['te_batch_size'])

    ##准备好模型
    model = mmformer_network(opt)
    model.cuda()
    # # model = torch.nn.DataParallel(model, device_ids=[0, 1])
    # parameters = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch_utils.get_optimizer(opt['optim'], parameters, opt['lr'])

    bestAcc = 0
    bestAcc0 = 0
    bestAcc1 = 0

    test_epoch = np.zeros(shape=(opt['num_epoch'], 4), dtype=float)
    for epoch in range(1, opt['num_epoch']+1):
        train_batch_num = Train_x_modal_2.shape[0] // opt['tr_batch_size']  # 1382
        test_batch_num = Test_x_modal_2.shape[0] // opt['te_batch_size']  # 7680/50=153
        Pre_epoch = np.zeros(shape=(Test_x_modal_2.shape[0], 2), dtype=float)
        YY_epoch = np.zeros(shape=(Test_x_modal_2.shape[0]), dtype=int)
        Test_batch = np.zeros(shape=(test_batch_num, 4), dtype=float)
        train_loss = 0
        train_acc = 0
        train_acc0 = 0
        train_acc1 = 0
        MyDataLoader = Data.DataLoader(MyDataSet_train(Train_x_modal_1, Train_x_modal_2, Train_y), opt['tr_batch_size'], num_workers=0, shuffle=True)
        b = 0
        for train_x_modal_1, train_x_modal_2, train_y in MyDataLoader:   ###Face, EEG
            if train_x_modal_2.shape[0] > opt['tr_batch_size'] and train_x_modal_2.shape[0] < opt['tr_batch_size']:
                continue      
            train_x_modal_2 = train_x_modal_2.cuda()
            train_x_modal_1 = train_x_modal_1.cuda()
            train_y = train_y.cuda()
            log, loss = model.train(train_x_modal_1, train_x_modal_2, train_y)
            logits = np.argmax(log.data.cpu().numpy(), axis=1)       

            acc = 0
            acc1 = 0
            acc2 = 0
            total1=0
            total2=0

            train_y = train_y.cpu()
            train_y = train_y.reshape(-1)
            train_y = np.array(train_y)
            for (y1, y2) in zip (logits, train_y):
                if y2 == 0:
                    total1 +=1
                if y2 == 1:
                    total2 +=1

                if y1 == y2:
                    acc += 1
                    if y2 ==0:
                        acc1 += 1
                    elif y2 == 1:
                        acc2 += 1

            Train_acc = float(acc) / (logits.shape[0])
            Train_acc0 = float(acc1) / total1
            Train_acc1 = float(acc2) / total2
            # print("epoch=", epoch,":", "batch =", b+1,":", "\n" , "train_acc_batch:", Train_acc, "train_acc1_batch:",Train_acc0, 
            #      "train_acc2_batch:", Train_acc1, "Train_loss_batch:", loss,  "\n")
            # b=b+1
            train_loss += loss
            train_acc += Train_acc
            train_acc0 += Train_acc0
            train_acc1 += Train_acc1

        # print("*************测试集**********************")
        k = 0
        MyDataLoader = Data.DataLoader(MyDataSet_test(Test_x_modal_1, Test_x_modal_2, Test_y), opt['te_batch_size'], num_workers=0, shuffle=True) 
        for test_x_modal_1, test_x_modal_2,  test_y in MyDataLoader:  

            test_x_modal_2 = test_x_modal_2.cuda()
            test_x_modal_1 = test_x_modal_1.cuda()
            test_y = test_y.cuda()

            if test_x_modal_2.shape[0] > opt['te_batch_size'] and test_x_modal_2.shape[0] < opt['te_batch_size']:
                continue
            # print("Testing on test dataset...")
            # predicts, loss, pred_out = model.predict(test_x_modal_1, test_x_modal_2, test_y)
            predicts, loss = model.predict(test_x_modal_1, test_x_modal_2, test_y)
            preds = np.argmax(predicts.data.cpu().numpy(), axis=1)
            test_y = test_y.cpu()
            test_y = test_y.reshape(-1)
            test_y = np.array(test_y)
            acc_t = 0
            acc1_t = 0
            acc2_t = 0

            total1_t= 0
            total2_t= 0


            for (y1, y2) in zip (preds, test_y):
                if y2 == 0:
                    total1_t +=1
                if y2 == 1:
                    total2_t +=1
                if y1 == y2:
                    acc_t += 1
                    if y2 ==0:
                        acc1_t += 1
                    elif y2 == 1:
                        acc2_t += 1
            Test_acc = float(acc_t) / (preds.shape[0])
            Test_acc0 = float(acc1_t) / total1_t
            Test_acc1 = float(acc2_t) / total2_t

            # print("epoch=", epoch,":", "batch =", k+1,":", "\n" , "test_acc_batch:", Test_acc,  
            #         "test_acc1_batch:",Test_acc0,  "test_acc2_batch:", Test_acc1,
            #         "Test_loss_batch:", loss.data.item(), "\n")
            ##########测试集的batch个数合到一起
            Test_batch[k,0] = Test_acc
            Test_batch[k,1] = Test_acc0
            Test_batch[k,2] = Test_acc1
            Test_batch[k,3] = loss

            Pre_epoch[242*k:242*(k+1),:] = pred_out.detach().cpu().numpy()
            YY_epoch[242*k:242*(k+1)] = preds

            k = k + 1

        # print ("**************训练集/测试集的epoch*************")  #一个epoch就是所有batch_size的平均值
        print('[epoch:%d] TR_Acc: %.3f%%, TR_Acc1: %.3f%%, TR_Acc2: %.3f%%,| Loss: %.03f || TE_Acc: %.3f%%, TE_Acc1: %.3f%%, TE_Acc2: %.3f%%,| Loss: %.03f  ' 
            %(epoch, 100. * train_acc / train_batch_num, 100. *train_acc0 / train_batch_num,
                100. * train_acc1 / train_batch_num, train_loss / train_batch_num, 100. * sum(Test_batch[:,0])/ test_batch_num, 
                100. * sum(Test_batch[:,1])/ test_batch_num, 100. * sum(Test_batch[:,2])/ test_batch_num, sum(Test_batch[:,3])/ test_batch_num))

        # print ("##############################################")
        ##################所有batch的平均值是一个epoch
        test_epoch [epoch-1,0] = sum(Test_batch[:,0]) / test_batch_num
        test_epoch [epoch-1,1] = sum(Test_batch[:,1]) / test_batch_num
        test_epoch [epoch-1,2] = sum(Test_batch[:,2]) / test_batch_num
        test_epoch [epoch-1,3] = sum(Test_batch[:,3]) / test_batch_num

    # import matplotlib.pyplot as plt
    # from sklearn.manifold import TSNE
    # import seaborn as sns
    # # 创建 t-SNE 对象
    # tsne = TSNE(n_components=2, perplexity=10)

    # # 对数据进行降维处理
    # X_tsne = tsne.fit_transform(Pre_epoch)

    # # 将降维结果可视化
    # # sns.scatterplot(x=X_tsne[:,0], y =X_tsne[:,1], hue= test_y, cmap='plasma', legend = False, palette='bright') 

    # plt.figure(figsize=(5,5)) # 单位是inches
    # current_axes=plt.axes()
    # current_axes.xaxis.set_visible(False)
    # current_axes.yaxis.set_visible(False)
    
    # # 将降维结果可视化
    # plt.scatter(x=X_tsne[:,0], y =X_tsne[:,1], c = YY_epoch, cmap='coolwarm', s = 30)
    # # sns.scatterplot(x=X_tsne[:,0], y=X_tsne[:,1], hue= YY_epoch, palette='bright')
    # plt.show()
    # plt.savefig('A_beta_20.jpg')

    
    # import matplotlib.pyplot as plt
    # from sklearn.manifold import TSNE
    # import seaborn as sns
    # # 创建 t-SNE 对象
    # tsne = TSNE(n_components=2, perplexity=10)

    # # 对数据进行降维处理
    # X_tsne = tsne.fit_transform(pred_out.detach().cpu().numpy())

    # # 将降维结果可视化
    # # sns.scatterplot(x=X_tsne[:,0], y =X_tsne[:,1], hue= test_y, cmap='plasma', legend = False, palette='bright') 
    # # plt.tick_params(labelleft=False, left=False)
    # # plt.tick_params(labelbottom=False, bottom=False)
    # plt.scatter(x=X_tsne[:,0], y =X_tsne[:,1], c = test_y, cmap='coolwarm')
    # plt.show()
    # plt.savefig('test2.jpg')


     ######所有epoch的平均就是一个fold
    te_result_epoch = np.mean(test_epoch, axis=0)
    te_result_batch = np.mean(Test_batch, axis=0)
    print("*********当前折---测试集结果*****************************")
    print( '[curr_fold: %d] | TE_Acc: %.3f%%, TE_Acc1: %.3f%%, TE_Acc2: %.3f%%' 
            % (cur_fold+1, te_result_epoch[0],  te_result_epoch[1],  te_result_epoch[2]))

    # print("Testing:", "curr_fold=", cur_fold+1,":", "\n", "last_test_acc:",  te_result_epoch[0], "last_test_acc1:", te_result_epoch[1], 
    #       "last_test_acc2:", te_result_epoch[2], "last_test_loss:", te_result_epoch[3], "\n")

    ########找出训练和测试的batch中最好的结果所对应的所有结果（列）
    te_row = np.argmax(Test_batch[:,0]) #找出所有batch中最好准确率的结果
    te_be_batch = Test_batch[te_row,:]
    ###########找出训练和测试的epoch中最好的结果所对应的所有结果（列）,所有batch的总计就是一个epoch
    te_row = np.argmax (test_epoch[:,0])
    te_be_epoch = test_epoch [te_row,:]
    print("Best_Test:", "curr_fold=", cur_fold+1,":", "Ba_te_acc:", te_be_batch, "\n", "Ep_te_acc:", te_be_epoch, "\n")

    times = "%s"%datetime.now()
    curr = cur_fold +1 
    Fold = "Fold[%d]"% curr 
    batch_acc = "%f" % te_result_batch[0]
    batch_acc1 = "%f" % te_result_batch[1]
    batch_acc2 = "%f" % te_result_batch[2]
    epoch_acc = "%f" % te_result_epoch[0]
    epoch_acc1 = "%f" % te_result_epoch[1]
    epoch_acc2 = "%f" % te_result_epoch[2]
    Bestbatch_acc = "%f" % te_be_batch[0]
    Bestbatch_acc0 = "%f" % te_be_batch[1]
    Bestbatch_acc1 = "%f" % te_be_batch[2]
    Bestepoch_acc = "%f" % te_be_epoch[0]
    Bestepoch_acc0 = "%f" % te_be_epoch[1]
    Bestepoch_acc1 = "%f" % te_be_epoch[2]
    Best_acc = "%f" %  bestAcc
    Best_acc0 = "%f" %  bestAcc0
    Best_acc1 = "%f" %  bestAcc1


    #########将数据结果保存到一维列表
    col = [times, curr, epoch_acc, epoch_acc1,epoch_acc2, Bestepoch_acc, Bestepoch_acc0, Bestepoch_acc1]
    Res = pandas.DataFrame([col])
    Res.to_csv(os.path.join(root_path,csv_name), mode= 'a' ,header=False, index=False)

    # bat += te_result_batch[0]
    # bat1 += te_result_batch[1]
    # bat2 += te_result_batch[2]

    # e += te_result_epoch[0]
    # e1 += te_result_epoch[1]
    # e2 += te_result_epoch[2]
# df = ['miss:0.0_w:1e-8_lam1:0.001_lam2:0.001']
# Res = Res.append(df, ignore_index = True)
# Res.to_csv(os.path.join(root_path,csv_name), mode= 'a' ,header=False, index=False)

# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# import seaborn as sns
# # 创建 t-SNE 对象
# tsne = TSNE(n_components=2, perplexity=10)

# # 对数据进行降维处理
# X_tsne = tsne.fit_transform(Pre_epoch.detach().cpu().numpy())

# # 将降维结果可视化
# # sns.scatterplot(x=X_tsne[:,0], y =X_tsne[:,1], hue= test_y, cmap='plasma', legend = False, palette='bright') 
# # plt.tick_params(labelleft=False, left=False)
# # plt.tick_params(labelbottom=False, bottom=False)
# current_axes=plt.axes()
# current_axes.xaxis.set_visible(False)
# current_axes.yaxis.set_visible(False)
# plt.rcParams['figure.figsize'] = (8.0, 8.0) # 单位是inches
# plt.scatter(x=X_tsne[:,0], y =X_tsne[:,1], c = Pre_epoch, cmap='coolwarm', s = 30)
# plt.show()
# plt.savefig('A_50_.jpg')

duration = time.time() - global_start_time
print("Duration time:", duration)