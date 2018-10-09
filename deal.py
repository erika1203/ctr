from scipy import sparse
path = '/users/lisa/Downloads/kedaxunfei/data'

base_train_csr = sparse.load_npz(path + '/base_train_csr.npz').tocsr().astype('bool')
base_predict_csr = sparse.load_npz(path + '/base_predict_csr.npz').tocsr().astype('bool')
train=base_train_csr.toarray()
print(train)
train=base_train_csr.toarray().astype('int32')
print(train)
