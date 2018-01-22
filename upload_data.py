import cPickle
import pandas as pd
from odps import ODPS

train_data, train_score, test_data = cPickle.load(open('output/0119/online_data_final.pkl'))
regressor, scaler = cPickle.load(open('output/0119/Lasso_Regressor.pkl'))
train_data = scaler.transform(train_data)
train_score = scaler.y_scaler.transform(train_score)
test_data = scaler.transform(test_data)

train_data.to_csv('explore/Lasso_train_data.csv')
test_data.to_csv('explore/Lasso_data.csv')

assert train_data.shape[0] == len(train_score)

train_data.columns = range(train_data.shape[1])

odps = ODPS('LTAIYQbRZMzJSs1V', 'DUHOB76E6mK4mm14o3NH2fD0r7im7y', 'YGTQ')

odps.delete_table('kv_train_data', if_exists=True)
odps.create_table('kv_train_data', 'append_id string,feature string, value double', if_not_exists=True)

records = []
for i in range(train_data.shape[0]):
	records.append((train_data.index[i],','.join(map(lambda x,y: str(x)+':'+str(y), train_data.columns,train_data.iloc[i, :])),
	                train_score[i]))

odps.write_table('kv_train_data', records)

test_data.columns = range(test_data.shape[1])
odps.delete_table('kv_test_data', if_exists=True)
odps.create_table('kv_test_data', 'append_id string,feature string', if_not_exists=True)

records = []
for i in range(test_data.shape[0]):
	records.append((test_data.index[i],','.join(map(lambda x,y: str(x)+':'+str(y), test_data.columns,test_data.iloc[i, :]))))

odps.write_table('kv_test_data', records)
