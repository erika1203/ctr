import datetime
from sklearn.feature_selection import chi2, SelectPercentile
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse
import lightgbm as lgb
import warnings
import time
import pandas as pd
import numpy as np
import os
import gc
from bayesSmooth import PH
from collections import Counter

path = '/users/lisa/Downloads/kedaxunfei/data'
threshold = 20

warnings.filterwarnings("ignore")

train = pd.read_table(path + '/round1_iflyad_train.txt')
test = pd.read_table(path + '/round1_iflyad_test_feature.txt')
data = pd.concat([train, test], axis=0, ignore_index=True)
# data = pd.read_csv(path + '/combining_data.csv',sep=',')
print('数据读取完毕.')

data = data.fillna(-1)

# 提取日、时特征
data['day'] = data['time'].apply(lambda x: int(time.strftime("%d", time.localtime(x))))
data['hour'] = data['time'].apply(lambda x: int(time.strftime("%H", time.localtime(x))))

# 分割广告行业特征
data['advert_industry_inner_1'] = data['advert_industry_inner'].apply(lambda x: x.split('_')[0])
data['ad_prefix']=data['adid'].apply(lambda x: int(str(x)[:3]))

#计算广告面积
data['area'] = data['creative_height'] * data['creative_width']

data['label'] = data.click.astype(int)
del data['click']
#
# # 将各特征中属于样本数少于20的类别合并为其他类
# for col in data.columns.values:
#     print(col,'combining')
#     if col in ["instance_id", "label", "user_tags","hour","day","time"]:
#         continue
#     x=data[col]
#     x_counter = Counter(x)
#     filter_num_list = list()
#     for item, cnt in x_counter.items():
#         if cnt > threshold:
#             continue
#         filter_num_list.append(item)
#         # print(item)
#     for item in filter_num_list:
#         data[col] = x.apply(lambda _item: -1 if _item == item else _item)
#         # print(item)
# print('少数类合并完毕.')
# data.to_csv(path+'/combining_data.csv',index=False)


# bool型特征
bool_feature = ['creative_is_jump', 'creative_is_download', 'creative_is_js', 'creative_is_voicead',
                'creative_has_deeplink', 'app_paid']
for i in bool_feature:
    data[i] = data[i].astype(int)

# 广告类别特征
ad_cate_feature = ['adid', 'advert_id', 'orderid', 'advert_industry_inner_1', 'advert_industry_inner', 'advert_name',
                   'campaign_id', 'creative_id', 'creative_type', 'creative_tp_dnf', 'creative_has_deeplink',
                   'creative_is_jump', 'creative_is_download','ad_prefix']

# 媒体类别特征
media_cate_feature = ['app_cate_id', 'f_channel', 'app_id', 'inner_slot_id']

# 上下文类别信息
content_cate_feature = ['city', 'carrier', 'province', 'nnt', 'devtype', 'osv', 'os', 'make', 'model']

origin_cate_list = ad_cate_feature + media_cate_feature + content_cate_feature

# 给各类别特征中不重复的值编号
for i in origin_cate_list:
    data[i] = data[i].map(dict(zip(data[i].unique(), range(0, data[i].nunique()))))
    # print(data[i])

cate_feature = origin_cate_list
print('类别特征不重复值编号完毕.')

rate_columns = ["province", "carrier", "devtype", "nnt", "os_name", "advert_id", "campaign_id", "creative_tp_dnf",
               "app_cate_id", "f_channel", "creative_type", "creative_is_jump", "creative_is_download",
               "creative_has_deeplink", "advert_name", "city", "adid", "orderid", "creative_id", "inner_slot_id",
               "app_id"]

# 获得平均点击率
# for col in rate_columns:
#     print(col,'rating')
#     x=data[col];y=data['label']
#     avg_rate=np.nanmean(y)
#     x_set = set(x)
#     rate_dict = dict()
#     for item in x_set:
#         sub_y = y[x == item]
#         if len(sub_y) <= threshold:
#             rate_dict[item] = avg_rate
#         sub_y_rate = np.nanmean(sub_y)
#         rate_dict[item] = sub_y_rate
#     column_name = "{column}_rate".format(column=col)
#     data[column_name] = data[col].map(rate_dict)
# print('点击率特征构造完毕.')
# data.to_csv(path+'/rating_data.csv',index=False)

# # 贝叶斯平滑
# PH_features=['advert_id', 'advert_industry_inner', 'advert_name', 'campaign_id', 'creative_height',
#                'creative_tp_dnf', 'creative_width','app_cate_id','osv','inner_slot_id']
# for fea in PH_features:
#     print(fea,'smoothing')
#     out=PH(fea,data)
#     # print(out);print(out.shape)
#     data=pd.merge(data,out,on=fea,how='left')

# 数值特征
num_feature = ['creative_width', 'creative_height', 'hour','area','day']
# num_feature = num_feature + ['PH_{}'.format(fea) for fea in PH_features]
# num_feature = num_feature + ['{}_rate'.format(col) for col in rate_columns]
feature = cate_feature + num_feature
print(len(feature), feature)
print('数值特征融合完毕.')

# 构造结果集
predict = data[data.label == -1]
predict_result = predict[['instance_id']]
predict_result['predicted_score'] = 0
predict_x = predict.drop('label', axis=1)

train_x = data[data.label != -1]
train_y = data[data.label != -1].label.values

# 压缩矩阵存储数据
# 默认加载 如果增加了cate类别特征 请改成false重新生成
if os.path.exists(path + '/base_train_csr.npz') and False:
    print('load_csr---------')
    base_train_csr = sparse.load_npz(path + '/base_train_csr.npz').tocsr().astype('bool')
    base_predict_csr = sparse.load_npz(path + '/base_predict_csr.npz').tocsr().astype('bool')
    print('压缩矩阵加载完毕.')
else:
    base_train_csr = sparse.csr_matrix((len(train), 0))
    base_predict_csr = sparse.csr_matrix((len(predict_x), 0))

    # 对类别特征进行onehot编码
    enc = OneHotEncoder()
    for feature in cate_feature:
        enc.fit(data[feature].values.reshape(-1, 1))
        base_train_csr = sparse.hstack((base_train_csr, enc.transform(train_x[feature].values.reshape(-1, 1))), 'csr',
                                       'bool')
        base_predict_csr = sparse.hstack((base_predict_csr, enc.transform(predict[feature].values.reshape(-1, 1))),
                                         'csr',
                                         'bool')
    print('one-hot prepared !')

    # 对多值属性进行词袋操作
    cv = CountVectorizer(min_df=20)
    for feature in ['user_tags']:
        data[feature] = data[feature].astype(str)
        cv.fit(data[feature])
        base_train_csr = sparse.hstack((base_train_csr, cv.transform(train_x[feature].astype(str))), 'csr', 'bool')
        base_predict_csr = sparse.hstack((base_predict_csr, cv.transform(predict_x[feature].astype(str))), 'csr',
                                         'bool')
    print('cv prepared !')

    sparse.save_npz(path + '/base_train_csr.npz', base_train_csr)
    sparse.save_npz(path + '/base_predict_csr.npz', base_predict_csr)
    print('压缩矩阵存储完毕.')

# 加入数值特征
train_csr = sparse.hstack(
    (sparse.csr_matrix(train_x[num_feature]), base_train_csr), 'csr').astype(
    'float32')
predict_csr = sparse.hstack(
    (sparse.csr_matrix(predict_x[num_feature]), base_predict_csr), 'csr').astype('float32')
print(train_csr.shape)

# 特征选择
feature_select = SelectPercentile(chi2, percentile=95)
feature_select.fit(train_csr, train_y)
train_csr = feature_select.transform(train_csr)
predict_csr = feature_select.transform(predict_csr)
print('特征选择完毕.')
print(train_csr.shape)

# 构造模型
lgb_model = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=32, reg_alpha=0, reg_lambda=0.1,
    max_depth=-1, n_estimators=5000, objective='binary',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
    learning_rate=0.05, random_state=2018, n_jobs=-1
)

# 五折交叉验证
skf = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)
best_score = []
for index, (train_index, test_index) in enumerate(skf.split(train_csr, train_y)):
    lgb_model.fit(train_csr[train_index], train_y[train_index],
                  eval_set=[(train_csr[train_index], train_y[train_index]),
                            (train_csr[test_index], train_y[test_index])], early_stopping_rounds=100)
    best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
    print(best_score)
    test_pred = lgb_model.predict_proba(predict_csr, num_iteration=lgb_model.best_iteration_)[:, 1]
    print('test mean:', test_pred.mean())
    predict_result['predicted_score'] = predict_result['predicted_score'] + test_pred
print(np.mean(best_score))
predict_result['predicted_score'] = predict_result['predicted_score'] / 5
mean = predict_result['predicted_score'].mean()
print('mean:', mean)
now = datetime.datetime.now()
now = now.strftime('%m-%d-%H-%M')
predict_result[['instance_id', 'predicted_score']].to_csv(path + "/result_sub/lgb_baseline_%s.csv" % now, index=False)