import scipy.special as special
from collections import Counter
import pandas as pd


class BayesianSmoothing(object):
    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def update(self, imps, clks, iter_num, epsilon):
        for i in range(iter_num):
            new_alpha, new_beta = self.__fixed_point_iteration(imps, clks, self.alpha, self.beta)
            if abs(new_alpha - self.alpha) < epsilon and abs(new_beta - self.beta) < epsilon:
                break
            # print(new_alpha, new_beta, i)
            self.alpha = new_alpha
            self.beta = new_beta

    def __fixed_point_iteration(self, imps, clks, alpha, beta):
        numerator_alpha = 0.0
        numerator_beta = 0.0
        denominator = 0.0
        for i in range(len(imps)):
            numerator_alpha += (special.digamma(clks[i] + alpha) - special.digamma(alpha))
            numerator_beta += (special.digamma(imps[i] - clks[i] + beta) - special.digamma(beta))
            denominator += (special.digamma(imps[i] + alpha + beta) - special.digamma(alpha + beta))

        return alpha * (numerator_alpha / denominator), beta * (numerator_beta / denominator)


def PH(feature,data):
    # -------------------------------------
    print('开始统计pos平滑')
    bs = BayesianSmoothing(1, 1)
    dic_i = dict(Counter(data[feature].values))
    dic_cov = dict(Counter(data[data['label'] == 1][feature].values))
    l = list(set(data[feature].values))
    I = []
    C = []
    for i in l:
        I.append(dic_i[i])
    for i in l:
        if i not in dic_cov:
            C.append(0)
        else:
            C.append(dic_cov[i])
    print('开始平滑操作')
    bs.update(I, C, 100000, 0.0000000001)
    print(bs.alpha, bs.beta)
    print('构建平滑转化率')
    dic_PH = {}
    for i in l:
        if i not in dic_i:
            dic_PH[i] = (bs.alpha) / (bs.alpha + bs.beta)
        elif i not in dic_cov:
            dic_PH[i] = (bs.alpha) / (dic_i[i] + bs.alpha + bs.beta)
        else:
            dic_PH[i] = (dic_cov[i] + bs.alpha) / (dic_i[i] + bs.alpha + bs.beta)
    out = pd.DataFrame({feature: list(dic_PH.keys()),
                        'PH_{}'.format(feature): list(dic_PH.values())})
    # print(df_out)
    # print(df_out.shape)
    # data=pd.merge(train, df_out, how='left', on=feature)
    # print(data)
    # print(data.shape)
    return out

# if __name__=='__main__':
#     path = '/users/lisa/Downloads/kedaxunfei/data'
#     train = pd.read_table(path + '/round1_iflyad_train.txt')
#     test = pd.read_table(path + '/round1_iflyad_test_feature.txt')
#     data = pd.concat([train, test], axis=0, ignore_index=True)
#     data=data.fillna(-1)
#     res=pd.DataFrame()
#     features=['osv','inner_slot_id','app_id','app_cate_id','make','model','advert_id','advert_industry_inner','advert_name','campaign_id', 'creative_height',
#                'creative_tp_dnf', 'creative_width', ]
#     for fea in features:
#         print(fea,'smoothing')
#         out=PH(fea,data)
#         print(out);print(out.shape)
#         # res=res.append(out,ignore_index=True)
#     # print(res);print(res.shape)




