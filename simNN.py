'''
2018-09-17
新材料企业数据
回归
'''

import torch
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import time
import numpy as np

from dataProcess import returnData
trainx,trainy,validx,validy,testx=returnData()



def train_batch(traind,trainl,SIZE=10,SHUFFLE=True,WORKER=2):   #分批处理
    trainset=Data.TensorDataset(traind,trainl)
    trainloader=Data.DataLoader(
        dataset=trainset,
        batch_size=SIZE,
        shuffle=SHUFFLE,
        num_workers=WORKER,  )
    return trainloader

# class Net(torch.nn.Module):
#     def __init__(self,n_feature,n_h1,n_h2,n_h3,n_output):
#         super(Net,self).__init__()
#         self.hidden1=torch.nn.Linear(n_feature,n_h1)
#         self.hidden2=torch.nn.Linear(n_h1,n_h2)
#         self.hidden3 = torch.nn.Linear(n_h2, n_h3)
#         self.predict=torch.nn.Linear(n_h3,n_output)
#
#     def forward(self, x):
#         x=F.relu(self.hidden1(x))
#         x = F.relu(self.hidden2(x))
#         x = F.relu(self.hidden3(x))
#         x=self.predict(x)
#         out = F.softmax(x, dim=1)
#         return out


EPOCH=1000
BATCH_SIZE=20
LR=0.001
MOMENTUM=0.9

train_loader=train_batch(trainx,trainy,BATCH_SIZE,SHUFFLE=True,WORKER=2)

net=torch.nn.Sequential(
    torch.nn.Linear(40,15),
    torch.nn.ReLU(),
    torch.nn.Linear(15,15),
    torch.nn.ReLU(),
    torch.nn.Linear(15,2),
 )

# torch.save(net,'net.pkl')
# torch.save(net.state_dict(),'net_params.pkl')

optimizer=torch.optim.SGD(net.parameters(),lr=LR,momentum=MOMENTUM)
loss_func=torch.nn.CrossEntropyLoss()

start_time = time.time()
for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        # print(x.data.numpy(),y.data.numpy())

        b_x=Variable(x)
        b_y=Variable(y)

        output=net(b_x).reshape(-1,)
        # print(prediction);print(output);print(b_y)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1000==0:
            valid_x=Variable(validx);test_y=Variable(validy)
            valid_out=net(valid_x).reshape(-1,)
            vloss=loss_func(valid_out,validy)
            # pre_val = torch.max(valid_out,1)[1].data.squeeze().numpy()
            # y_val = test_y.data.squeeze().numpy()
            # print(pre_val);print(y_val)
            # accuracy = float((pre_val == y_val).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, 'step:',step,'| train loss: %.4f' % loss.data.numpy(),
                  '| valid loss: %.2f' % vloss.data.numpy())
duration = time.time() - start_time
# print('Duration:%.4f' % duration)






















