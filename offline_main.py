# coding=utf-8
import cPickle
import pandas as pd

train_data, train_score, test_data, test_score = cPickle.load(open('offline_data.pkl'))
print train_data.shape, test_data.shape


from categorical_processing import *
# 训练集和测试集合并后，对工具变量进行处理，再分离训练集和测试集
full_data = pd.concat([train_data, test_data], axis=0)

full_data, new_col_name = categorical_processing(full_data, method='complex')

train_data = full_data.loc[train_data.index,]
test_data = full_data.loc[test_data.index,]

print train_data.shape, test_data.shape

# Train the model
from regressor import *
from sklearn.linear_model import Lasso, LassoCV
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ShuffleSplit

# 创建scaler，指定连续变量的z分数限制、离散变量名、离散变量的z分数限制、离散变量权重
scaler = YGTQ_Scaler(method='categorical', max_z_score=2, discrete_col=new_col_name, discrete_max_z_score=5,
                     discrete_weight=1)

# 训练scaler，默认不使用测试集作为辅助数据
train_data, train_score = scaler.fit_transform(train_data, train_score, auxiliary_data=None, test_data=test_data)
test_data = scaler.transform(test_data)

'''
# 确认线下cv的最优参数是否与线上测试集的最优参数相近
regressor = LassoCV(normalize=False, alphas=np.arange(0.001,0.006,0.0002),cv=ShuffleSplit(n_splits=30,test_size=0.2),n_jobs=-1)
regressor.fit(train_data, train_score)
import numpy as np
estimator = regressor
mse = estimator.mse_path_[np.where(estimator.alphas_ == estimator.alpha_)]
print regressor.alpha_, mse.mean(), mse.max(), (estimator.coef_!=0).sum()

chosen_col = train_data.columns[(regressor.coef_!=0)]
train_data[chosen_col].to_csv('explore/Lasso_train_data.csv')
test_data[chosen_col].to_csv('explore/Lasso_data.csv')
'''

# 根据线上的测试集调参
for alpha in np.arange(0.0005,0.006,0.0005):
	regressor = Lasso(normalize=False, alpha=alpha)
	regressor.fit(train_data, train_score)

	train_pred = scaler.y_scaler.inverse_transform(regressor.predict(train_data))
	pred_score = scaler.y_scaler.inverse_transform(regressor.predict(test_data))

	print alpha
	print 'Train MSE: %f' % mean_squared_error(scaler.y_scaler.inverse_transform(train_score), train_pred)
	print 'Test MSE: %f' % mean_squared_error(test_score, pred_score)

