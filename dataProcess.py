'''
2018-09-23
科大讯飞数据
simNN
'''

import torch
import torchvision
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
import time
import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder


def unify(func,data,):
    m, n = data.shape
    if func=='dif':
        maxMat = np.max(data, 0)
        minMat = np.min(data, 0)
        diffMat = maxMat - minMat
        for j in range(n - 1):
            data[:, j] = (data[:, j] - minMat[j]) / diffMat[j]

    if func=='std':
        for j in range(n-1):
            meanVal=np.mean(data[:,j])
            stdVal=np.std(data[:,j])
            data[:,j]=(data[:,j]-meanVal)/stdVal
    print(type(data),data.shape);print(data[0])
    return data

def data_process():     #数据预处理

    # 加载数据
    train = pd.read_table('./data/round1_iflyad_train.txt')
    test = pd.read_table('./data/round1_iflyad_test_feature.txt')

    # 合并训练集，验证集
    data = pd.concat([train, test], axis=0, ignore_index=True)

    # 缺失值填充
    data['make'] = data['make'].fillna(str(-1))
    data['model'] = data['model'].fillna(str(-1))
    data['osv'] = data['osv'].fillna(str(-1))
    data['app_cate_id'] = data['app_cate_id'].fillna(-1)
    data['app_id'] = data['app_id'].fillna(-1)
    data['click'] = data['click'].fillna(-1)
    data['user_tags'] = data['user_tags'].fillna(str(-1))
    data['f_channel'] = data['f_channel'].fillna(str(-1))

    # replace
    replace = ['creative_is_jump', 'creative_is_download', 'creative_is_js', 'creative_is_voicead',
               'creative_has_deeplink', 'app_paid']
    for feat in replace:
        data[feat] = data[feat].replace([False, True], [0, 1])

    # labelencoder 转化
    encoder = ['city', 'province', 'make', 'model', 'osv', 'os_name', 'adid', 'advert_id', 'orderid',
               'advert_industry_inner', 'campaign_id', 'creative_id', 'app_cate_id',
               'app_id', 'inner_slot_id', 'advert_name', 'f_channel', 'creative_tp_dnf']

    # 简化编码
    col_encoder = LabelEncoder()
    for feat in encoder:
        col_encoder.fit(data[feat])
        data[feat] = col_encoder.transform(data[feat])

    # 将秒数转换为{年，月，日，时，分，秒，周，年中第几天，是否夏令时}
    # 然后取出日和时
    data['day'] = data['time'].apply(lambda x: int(time.strftime("%d", time.localtime(x))))
    data['hour'] = data['time'].apply(lambda x: int(time.strftime("%H", time.localtime(x))))

    # 历史点击率
    # 时间转换，将5.27到6.3号转化为序列{27,28,29,30,31,32,33,34}，增加为period列
    data['period'] = data['day']
    data['period'][data['period'] < 27] = data['period'][data['period'] < 27] + 31

    for feat_1 in ['advert_id', 'advert_industry_inner', 'advert_name', 'campaign_id', 'creative_height',
                   'creative_tp_dnf', 'creative_width', 'province', 'f_channel']:
        # 垃圾回收
        gc.collect()

        res = pd.DataFrame()
        temp = data[[feat_1, 'period', 'click']]

        # 得到样本各特征的历史点击率，第一天为当天值，其他为当天之前的值的和
        for period in range(27, 35):
            if period == 27:
                # 该feat各值出现在数据中的次数
                count = temp.groupby([feat_1]).apply(
                    lambda x: x['click'][(x['period'] <= period).values].count()).reset_index(name=feat_1 + '_all')
                # 该feat各值被点击的次数
                count1 = temp.groupby([feat_1]).apply(
                    lambda x: x['click'][(x['period'] <= period).values].sum()).reset_index(name=feat_1 + '_1')
            else:
                count = temp.groupby([feat_1]).apply(
                    lambda x: x['click'][(x['period'] < period).values].count()).reset_index(name=feat_1 + '_all')
                count1 = temp.groupby([feat_1]).apply(
                    lambda x: x['click'][(x['period'] < period).values].sum()).reset_index(name=feat_1 + '_1')

            # 将上面的count和count1拼在一起，并填补缺失值
            count[feat_1 + '_1'] = count1[feat_1 + '_1']
            count.fillna(value=0, inplace=True)

            # 计算点击率
            count[feat_1 + '_rate'] = pd.Series([round(x, 5) for x in count[feat_1 + '_1'] / count[feat_1 + '_all']])
            count['period'] = period
            # 只保留点击率这一列
            count.drop([feat_1 + '_all', feat_1 + '_1'], axis=1, inplace=True)
            count.fillna(value=0, inplace=True)
            res = res.append(count, ignore_index=True)
        print(feat_1, ' over')

        # 左连接合并，数据中多出7个特征的历史点击率
        data = pd.merge(data, res, how='left', on=[feat_1, 'period'])

    # 删除没用的特征
    drop = ['click', 'time', 'instance_id', 'user_tags',
             'app_paid', 'creative_is_js', 'creative_is_voicead']


    train = data[:train.shape[0]]
    test = data[train.shape[0]:]
    y_train = train.loc[:, 'click']
    res = test.loc[:, ['instance_id']]

    train.drop(drop, axis=1, inplace=True)
    # print(type(train.values));print(train.values)
    X_loc_train = unify('dif',train.values)
    test.drop(drop, axis=1, inplace=True)
    y_loc_train = y_train.values
    X_loc_test = unify('dif',test.values)

    return X_loc_train,y_loc_train,X_loc_test

def gen_train_test(data):
    np.random.shuffle(data)
    traind=data[:700000,:-1]
    trainl=data[:700000,-1]
    validd=data[700000:,:-1]
    validl=data[700000:,-1]

    traind = torch.from_numpy(traind).float()
    trainl = torch.from_numpy(trainl).long()
    validd = torch.from_numpy(validd).float()
    validl = torch.from_numpy(validl).long()
    return traind,trainl,validd,validl


def train_batch(traind,trainl,SIZE=10,SHUFFLE=True,WORKER=2):   #分批处理
    trainset=Data.TensorDataset(traind,trainl)
    trainloader=Data.DataLoader(
        dataset=trainset,
        batch_size=SIZE,
        shuffle=SHUFFLE,
        num_workers=WORKER,  )
    return trainloader

def returnData():
    trainx,trainy,testx=data_process()
    trainx,trainy,validx,validy=gen_train_test(np.c_[trainx,trainy])
    print(trainx.shape,trainy.shape,validx.shape,validy.shape)
    return trainx,trainy,validx,validy,testx

