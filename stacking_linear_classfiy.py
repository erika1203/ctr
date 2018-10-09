import pandas as pd
from scipy import sparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC

train = pd.read_csv('data/round1_iflyad_train.txt', sep='\t')
test = pd.read_csv('data/round1_iflyad_test_feature.txt', sep='\t')

feature = sparse.load_npz('data/onehot_feature.npz')
print(feature.shape)

train_feature = feature[:len(train)]
test_feature = feature[len(train):]
score = train['click']

number = len(np.unique(score))

# 五则交叉验证
n_folds = 5
df_stack = pd.DataFrame()
df_stack['instance_id']=pd.concat([train, test], axis=0, ignore_index=True)['instance_id']
print('处理完毕')

########################### lr(LogisticRegression) ################################
print('lr stacking')
stack_train = np.zeros((len(train), number))
# print(number);print(stack_train.shape)
stack_test = np.zeros((len(test), number))
score_va = 0

skf=StratifiedKFold(n_splits=n_folds, random_state=1017)
for i, (tr, va) in enumerate(skf.split(train_feature,score)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    clf = LogisticRegression(random_state=1017, C=8)
    clf.fit(train_feature[tr], score[tr])
    score_va = clf.predict_proba(train_feature[va])
    # print(score_va.shape);print(score_va)
    score_te = clf.predict_proba(test_feature)
    print('得分' + str(log_loss(score[va], clf.predict(train_feature[va]))))
    stack_train[va] += score_va
    stack_test += score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])
# print(stack.shape);print(stack)

for i in range(stack_test.shape[1]):
    df_stack['lr_classfiy_{}'.format(i)] = np.around(stack[:, i], 6)
del df_stack['lr_classfiy_1']
print('lr特征已保存\n')

########################### SGD(随机梯度下降) ################################
print('sgd stacking')
stack_train = np.zeros((len(train), number))
stack_test = np.zeros((len(test), number))
score_va = 0

skf=StratifiedKFold(n_splits=n_folds,random_state=1017)
for i, (tr, va) in enumerate(skf.split(train_feature,score)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    sgd = SGDClassifier(random_state=1017, loss='log')
    sgd.fit(train_feature[tr], score[tr])
    score_va = sgd.predict_proba(train_feature[va])
    score_te = sgd.predict_proba(test_feature)
    print('得分' + str(log_loss(score[va], sgd.predict(train_feature[va]))))
    stack_train[va] += score_va
    stack_test += score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])
for i in range(stack_test.shape[1]):
    df_stack['sgd_classfiy_{}'.format(i)] = np.around(stack[:, i], 6)
del df_stack['sgd_classfiy_1']
print('sgd特征已保存\n')

########################### pac(PassiveAggressiveClassifier) ################################
print('PAC stacking')
stack_train = np.zeros((len(train), number))
stack_test = np.zeros((len(test), number))
score_va = 0

skf=StratifiedKFold(n_splits=n_folds,random_state=1017)
for i, (tr, va) in enumerate(skf.split(train_feature,score)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    pac = PassiveAggressiveClassifier(random_state=1017)
    pac.fit(train_feature[tr], score[tr])
    score_va = pac._predict_proba_lr(train_feature[va])
    score_te = pac._predict_proba_lr(test_feature)
    print(score_va)
    print('得分' + str(log_loss(score[va], pac.predict(train_feature[va]))))
    stack_train[va] += score_va
    stack_test += score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])
for i in range(stack_test.shape[1]):
    df_stack['pac_classfiy_{}'.format(i)] = np.around(stack[:, i], 6)
del df_stack['pac_classfiy_1']
print('pac特征已保存\n')


########################### ridge(RidgeClassfiy) ################################
print('RidgeClassfiy stacking')
stack_train = np.zeros((len(train), number))
stack_test = np.zeros((len(test), number))
score_va = 0

skf=StratifiedKFold(n_splits=n_folds,random_state=1017)
for i, (tr, va) in enumerate(skf.split(train_feature,score)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    ridge = RidgeClassifier(random_state=1017)
    ridge.fit(train_feature[tr], score[tr])
    score_va = ridge._predict_proba_lr(train_feature[va])
    score_te = ridge._predict_proba_lr(test_feature)
    print(score_va)
    print('得分' + str(log_loss(score[va], ridge.predict(train_feature[va]))))
    stack_train[va] += score_va
    stack_test += score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])
for i in range(stack_test.shape[1]):
    df_stack['ridge_classfiy_{}'.format(i)] = np.around(stack[:, i], 6)
del df_stack['ridge_classfiy_1']
print('ridge特征已保存\n')


########################### bnb(BernoulliNB) ################################
print('BernoulliNB stacking')
stack_train = np.zeros((len(train), number))
stack_test = np.zeros((len(test), number))
score_va = 0

skf=StratifiedKFold(n_splits=n_folds,random_state=1017)
for i, (tr, va) in enumerate(skf.split(train_feature,score)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    bnb = BernoulliNB()
    bnb.fit(train_feature[tr], score[tr])
    score_va = bnb.predict_proba(train_feature[va])
    score_te = bnb.predict_proba(test_feature)
    print(score_va)
    print('得分' + str(log_loss(score[va], bnb.predict(train_feature[va]))))
    stack_train[va] += score_va
    stack_test += score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])
for i in range(stack_test.shape[1]):
    df_stack['bnb_classfiy_{}'.format(i)] = np.around(stack[:, i], 6)
del df_stack['bnb_classfiy_1']
print('BernoulliNB特征已保存\n')

########################### mnb(MultinomialNB) ################################
print('MultinomialNB stacking')
stack_train = np.zeros((len(train), number))
stack_test = np.zeros((len(test), number))
score_va = 0

skf=StratifiedKFold(n_splits=n_folds,random_state=1017)
for i, (tr, va) in enumerate(skf.split(train_feature,score)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    mnb = MultinomialNB()
    mnb.fit(train_feature[tr], score[tr])
    score_va = mnb.predict_proba(train_feature[va])
    score_te = mnb.predict_proba(test_feature)
    print(score_va)
    print('得分' + str(log_loss(score[va], mnb.predict(train_feature[va]))))
    stack_train[va] += score_va
    stack_test += score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])
for i in range(stack_test.shape[1]):
    df_stack['mnb_classfiy_{}'.format(i)] = np.around(stack[:, i], 6)
del df_stack['mnb_classfiy_1']
print('MultinomialNB特征已保存\n')

############################ Linersvc(LinerSVC) ################################
print('LinerSVC stacking')
stack_train = np.zeros((len(train), number))
stack_test = np.zeros((len(test), number))
score_va = 0

skf=StratifiedKFold(n_splits=n_folds,random_state=1017)
for i, (tr, va) in enumerate(skf.split(train_feature,score)):
    print('stack:%d/%d' % ((i + 1), n_folds))
    lsvc = LinearSVC(random_state=1017)
    lsvc.fit(train_feature[tr], score[tr])
    score_va = lsvc._predict_proba_lr(train_feature[va])
    score_te = lsvc._predict_proba_lr(test_feature)
    print(score_va)
    print('得分' + str(log_loss(score[va], lsvc.predict(train_feature[va]))))
    stack_train[va] += score_va
    stack_test += score_te
stack_test /= n_folds
stack = np.vstack([stack_train, stack_test])
for i in range(stack_test.shape[1]):
    df_stack['lsvc_classfiy_{}'.format(i)] = np.around(stack[:, i], 6)
del df_stack['lsvc_classfiy_1']
print('LSVC特征已保存\n')


###### kmeans ###
# def get_cluster(num_clusters):
#     print('开始' + str(num_clusters))
#     name = 'kmean'
#     print(name)
#     model = KMeans(n_clusters=num_clusters, max_iter=300, n_init=1, \
#                         init='k-means++', n_jobs=10, random_state=1017)
#     result = model.fit_predict(feature)
#     df_stack[name + 'word_' + str(num_clusters)] = result
#
# get_cluster(5)
# get_cluster(10)
# get_cluster(19)
# get_cluster(30)
# get_cluster(40)
# get_cluster(50)
# get_cluster(60)
# get_cluster(70)

df_stack.to_csv('data/stacking_linear_classfiy_all.csv',index=False)