import pandas as pd
from odps import ODPS

odps = ODPS('LTAIYQbRZMzJSs1V', 'DUHOB76E6mK4mm14o3NH2fD0r7im7y', 'YGTQ')

result = odps.read_table('test_predict')
output = pd.DataFrame(map(lambda x: [x.values[0],x.values[-2]], result))
output.iloc[:,0] = output.iloc[:,0].apply(lambda x: x.strip('_new'))
output.to_csv('output_A_pai.csv',header=None,index=None)