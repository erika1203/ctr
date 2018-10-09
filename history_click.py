import pandas as pd
import gc

# 计算历史点击率
train = pd.read_csv('/data/round1_iflyad_train.txt', sep='\t')
test = pd.read_table('/data/round1_iflyad_test_feature.txt', sep='\t')
data = pd.concat([train, test], axis=0, ignore_index=True)
data=data.fillna(-1)

data['period'] = data['day']
data['period'][data['period']<27] = data['period'][data['period']<27] + 31
for feat_1 in ['advert_id','advert_industry_inner','advert_name','campaign_id', 'creative_height',
               'creative_tp_dnf', 'creative_width', 'province', 'f_channel']:
    # 垃圾回收
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

        # 计算点击率
        count[feat_1+'_rate'] = pd.Series([round(x,5) for x in count[feat_1 + '_1'] / count[feat_1 + '_all']])
        count['period']=period

        # 只保留点击率这一列
        count.drop([feat_1+'_all', feat_1+'_1'],axis=1,inplace=True)
        count.fillna(value=0, inplace=True)
        res=res.append(count,ignore_index=True)
    print(feat_1,' over')
res.to_csv('/data/history_click.csv',index=False)