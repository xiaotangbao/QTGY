import pandas as pd
from odps import ODPS
import cPickle

odps = ODPS('LTAIYQbRZMzJSs1V', 'DUHOB76E6mK4mm14o3NH2fD0r7im7y', 'YGTQ')
regressor, scaler = cPickle.load(open('output/0119/Lasso_Regressor.pkl'))

result = odps.read_table('test_predict')
output = pd.DataFrame(map(lambda x: [x.values[0],x.values[-2]], result),columns=['ID','value'])
output['value'] = scaler.y_scaler.inverse_transform(output['value'])
output['ID'] = output['ID'].apply(lambda x: x.strip('_new'))
output.to_csv('output_A_pai.csv',header=None,index=None)