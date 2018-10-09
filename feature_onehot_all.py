import pandas as pd
import scipy.sparse
import time
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gc

train = pd.read_csv('data/round1_iflyad_train.txt', sep='\t')
test = pd.read_csv('data/round1_iflyad_test_feature.txt', sep='\t')
data = pd.concat([train, test],sort=False,axis=0)

### time字段 月日时 onehot
data['time_hour'] = data['time'].apply(lambda row: int(time.localtime(row).tm_hour))
data['time_day'] = data['time'].apply(lambda row: int(time.localtime(row).tm_mday))
data['time_month'] = data['time'].apply(lambda row: int(time.localtime(row).tm_mon))

hour_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['time_hour']))
day_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['time_day']))
month_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['time_month']))
onehot_feature = scipy.sparse.hstack([hour_deal, day_deal, month_deal])
print('第一组完毕')
gc.collect()

### city,province
city_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['city']))
province_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['province']))
onehot_feature = scipy.sparse.hstack([onehot_feature, city_deal, province_deal])
print('第二组完毕')
gc.collect()

### user_tags(tfidf,count)
data['user_tags'] = data['user_tags'].fillna('0')
data['user_tags'] = data['user_tags'].apply(lambda row:str(row).replace(',', ' '))
count_vec = CountVectorizer()
count_csr_basic = scipy.sparse.csr_matrix(count_vec.fit_transform(data['user_tags']))
tfidf_vec = TfidfVectorizer()
tfidf_vec_basic = scipy.sparse.csr_matrix(tfidf_vec.fit_transform(data['user_tags']))
onehot_feature = scipy.sparse.hstack([onehot_feature, count_csr_basic, tfidf_vec_basic])
print('第三组完毕')
gc.collect()

### 'carrier','devtype', 'make', 'model', 'nnt', 'os', 'osv', 'os_name'
carrier_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['carrier']))
devtype_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['devtype']))
make_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['make']))
model_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['model']))
nnt_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['nnt']))
os_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['os']))
osv_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['osv']))
os_name_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['os_name']))
onehot_feature = scipy.sparse.hstack([onehot_feature, carrier_deal, devtype_deal,
                                                              make_deal, model_deal, nnt_deal, os_deal,
                                                              osv_deal, os_name_deal])
print('第四组完毕')
gc.collect()

### 'adid', 'advert_id', 'orderid'
adid_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['adid']))
advert_id_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['advert_id']))
orderid_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['orderid']))
onehot_feature = scipy.sparse.hstack([onehot_feature, adid_deal, advert_id_deal, orderid_deal])
print('第五组完毕')
gc.collect()

### 'advert_industry_inner'
data['aii_1'] = data['advert_industry_inner'].apply(lambda row:row.split('_')[0])
data['aii_2'] = data['advert_industry_inner'].apply(lambda row:row.split('_')[1])
aii_1_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['aii_1']))
aii_2_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['aii_2']))
onehot_feature = scipy.sparse.hstack([onehot_feature, aii_1_deal, aii_2_deal])
print('第六组完毕')
gc.collect()

### 'campaign_id', 'creative_id', 'creative_tp_dnf', 'app_cate_id', 'f_channel', 'app_id', 'inner_slot_id'
### 'creative_type', 'creative_width', 'creative_height','creative_is_jump', 'creative_is_download', 'creative_is_js',
### 'creative_is_voicead', 'creative_has_deeplink'
campaign_id_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['campaign_id']))
creative_id_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['creative_id']))
creative_tp_dnf_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['creative_tp_dnf']))
app_cate_id_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['app_cate_id']))
f_channel_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['f_channel']))
app_id_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['app_id']))
inner_slot_id_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['inner_slot_id']))
creative_type_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['creative_type']))
creative_width_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['creative_width']))
creative_height_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['creative_height']))
creative_is_jump_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['creative_is_jump']))
creative_is_download_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['creative_is_download']))
creative_is_js_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['creative_is_js']))
creative_is_voicead_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['creative_is_voicead']))
creative_has_deeplink_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['creative_has_deeplink']))
onehot_feature = scipy.sparse.hstack([onehot_feature, campaign_id_deal, creative_id_deal,
                                                              creative_tp_dnf_deal, app_cate_id_deal, f_channel_deal,
                                                              app_id_deal, inner_slot_id_deal, creative_type_deal,
                                                              creative_width_deal, creative_height_deal, creative_is_jump_deal,
                                                              creative_is_download_deal, creative_is_js_deal, creative_is_voicead_deal,
                                                              creative_has_deeplink_deal])
print('第七组完毕')
gc.collect()

### 'app_paid', 'advert_name'
app_paid_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['app_paid']))
advert_name_deal = scipy.sparse.csr_matrix(pd.get_dummies(data['advert_name']))
onehot_feature = scipy.sparse.hstack([onehot_feature, app_id_deal, advert_name_deal])
print('第八组完毕')
gc.collect()

onehot_feature = scipy.sparse.csr_matrix(onehot_feature)

scipy.sparse.save_npz('data/onehot_feature.npz', onehot_feature)  #保存

# csr_matrix_variable = sparse.load_npz('path.npz') #读
