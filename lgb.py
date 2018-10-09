'''
2018-09-21
试跑知乎大佬的代码
'''


import numpy as np
import pandas as pd
import time
import datetime
import gc
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import lightgbm as lgb
import sparse
import re



# 加载数据
train = pd.read_table('./data/round1_iflyad_train.txt')
test = pd.read_table('./data/round1_iflyad_test_feature.txt')

# 合并训练集，验证集
data = pd.concat([train,test],axis=0,ignore_index=True)

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
replace = ['creative_is_jump', 'creative_is_download', 'creative_is_js', 'creative_is_voicead', 'creative_has_deeplink', 'app_paid']
for feat in replace:
    data[feat] = data[feat].replace([False, True], [0, 1])

#增加广告前缀特征
data['ad_prefix']=data['adid'].apply(lambda x: int(str(x)[:2]))

#将行业主次分开
data['advert_industry_inner_first'] = data['advert_industry_inner'].apply(lambda x:x.split('_')[0])
data['advert_industry_inner_second'] = data['advert_industry_inner'].apply(lambda x:x.split('_')[1])
data.drop('advert_industry_inner',axis=1,inplace=True)

#构造广告面积特征
data['area'] = data['creative_height'] * data['creative_width']

# labelencoder 转化
encoder = ['city', 'province', 'make', 'model', 'osv', 'os_name', 'adid', 'advert_id', 'orderid',
            'campaign_id', 'creative_id', 'app_cate_id','app_id', 'inner_slot_id', 'advert_name', 'area',
           'f_channel', 'creative_tp_dnf','ad_prefix','advert_industry_inner_first','advert_industry_inner_second']

#简化编码
col_encoder = LabelEncoder()
for feat in encoder:
    col_encoder.fit(data[feat])
    data[feat] = col_encoder.transform(data[feat])

#将秒数转换为{年，月，日，时，分，秒，周，年中第几天，是否夏令时}
#然后取出日和时
data['day'] = data['time'].apply(lambda x : int(time.strftime("%d", time.localtime(x))))
data['hour'] = data['time'].apply(lambda x : int(time.strftime("%H", time.localtime(x))))

# 历史点击率
# 时间转换，将5.27到6.3号转化为序列{27,28,29,30,31,32,33,34}，增加为period列
data['period'] = data['day']
data['period'][data['period']<27] = data['period'][data['period']<27] + 31

# data_time_min = data['time'].min()
# data['day'] = (data['time'].values - data_time_min) / 3600 / 24
# data['day'] = data['day'].astype(int)
# data['hour'] = (data['time'].values - data_time_min - data['day'].values * 3600 * 24) / 3600
# data['hour'] = data['hour'].astype(int)

for feat_1 in ['advert_id','advert_industry_inner_first','advert_industry_inner_second','advert_name','campaign_id', 'creative_height',
               'creative_tp_dnf', 'creative_width', 'province', 'f_channel']:
    #垃圾回收
    gc.collect()

    res=pd.DataFrame()
    temp=data[[feat_1,'period','click']]

    # 得到样本各特征的历史点击率，第一天为当天值，其他为当天之前的值的和
    for period in range(27,35):
        if period == 27:
            # 该feat各值出现在数据中的次数
            count=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<=period).values].count()).reset_index(name=feat_1+'_all')
            # 该feat各值被点击的次数
            count1=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<=period).values].sum()).reset_index(name=feat_1+'_1')
        else:
            count=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<period).values].count()).reset_index(name=feat_1+'_all')
            count1=temp.groupby([feat_1]).apply(lambda x: x['click'][(x['period']<period).values].sum()).reset_index(name=feat_1+'_1')

        # 将上面的count和count1拼在一起，并填补缺失值
        count[feat_1+'_1']=count1[feat_1+'_1']
        count.fillna(value=0, inplace=True)

        #计算点击率
        count[feat_1+'_rate'] = pd.Series([round(x,5) for x in count[feat_1+'_1'] / count[feat_1+'_all']])
        count['period']=period
        #只保留点击率这一列
        count.drop([feat_1+'_all', feat_1+'_1'],axis=1,inplace=True)
        count.fillna(value=0, inplace=True)
        res=res.append(count,ignore_index=True)
    print(feat_1,' over')

    # 左连接合并，数据中多出9个特征的历史点击率
    data = pd.merge(data,res, how='left', on=[feat_1,'period'])


# 删除没用的特征
drop = ['click', 'time', 'instance_id', 'user_tags','creative_is_voicead','creative_is_js','app_paid']


train = data[:train.shape[0]]
test = data[train.shape[0]:]
y_train = train.loc[:,'click']
res = test.loc[:, ['instance_id']]

train.drop(drop, axis=1, inplace=True)
print('train:',train.shape)
test.drop(drop, axis=1, inplace=True)
print('test:',test.shape)

#转化为数组
X_loc_train = train.values
y_loc_train = y_train.values
print(y_loc_train.tolist().count(1.))
X_loc_test = test.values


# 模型部分
model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=48, max_depth=-1, learning_rate=0.03, n_estimators=2000,
                           max_bin=425, subsample_for_bin=50000, objective='binary', min_split_gain=0,
                           min_child_weight=5, min_child_samples=10, subsample=0.8, subsample_freq=1,
                           colsample_bytree=1, reg_alpha=3, reg_lambda=5, seed=1000, n_jobs=10, silent=True)

# 五折交叉训练，构造五个模型
skf=list(StratifiedKFold(y_loc_train, n_folds=5, shuffle=True, random_state=1024))
baseloss = []
loss = 0
for i, (train_index, test_index) in enumerate(skf):
    #训练
    print("Fold", i)
    lgb_model = model.fit(X_loc_train[train_index], y_loc_train[train_index],
                          eval_names =['train','valid'],
                          eval_metric='logloss',
                          eval_set=[(X_loc_train[train_index], y_loc_train[train_index]),
                                    (X_loc_train[test_index], y_loc_train[test_index])],early_stopping_rounds=100)
    #训练损失
    baseloss.append(lgb_model.best_score_['valid']['binary_logloss'])
    loss += lgb_model.best_score_['valid']['binary_logloss']


    test_pred= lgb_model.predict_proba(X_loc_test, num_iteration=lgb_model.best_iteration_)[:, 1]
    print('test mean:', test_pred.mean())
    res['prob_%s' % str(i)] = test_pred
print('logloss:', baseloss, loss/5)

# 加权平
res['predicted_score'] = 0
for i in range(5):
    res['predicted_score'] += res['prob_%s' % str(i)]
res['predicted_score'] = res['predicted_score']/5

# 提交结果
mean = res['predicted_score'].mean()
print('mean:',mean)
now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
res[['instance_id', 'predicted_score']].to_csv("./result_sub/lgb_baseline_%s.csv" % now, index=False)




