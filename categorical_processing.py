# coding=utf-8
import pandas as pd
import re

def select_categorical(data, category_n):
    import re
    tool_mark = re.compile(r'[A-Za-z]+_?[A-Za-z]+.*')
    tool_columns = filter(lambda x: re.match(tool_mark, x), data.columns)
    categorical_columns = []
    for i in set(data.columns) - set(tool_columns):
        if (len(set(data.loc[:, i])) <= category_n):
            categorical_columns.append(i)
    return categorical_columns

def _identify_categorical_variable(df):
    col = ['TOOL', 'Tool', 'TOOL_ID', 'Tool (#1)', 'TOOL (#1)', 'TOOL (#2)', 'Tool (#2)', 'Tool (#3)', 'Tool (#4)',
     'OPERATION_ID','Tool (#5)', 'TOOL (#3)']
    assert set(col) < set(df.columns)
    return col

def categorical_encoding(data, col_name, encoder=None,mapping_dict=None,new_col_name=None, test_data=None):

    from sklearn.preprocessing import OneHotEncoder
    def concat_encoder(data,col_name,categorical_data,new_col_name):
        # 使用离散化后的变量替代原有的分类变量，新变量放在DataFrame的最后
        assert categorical_data.shape[0] == data.shape[0]
        assert categorical_data.shape[1] == len(new_col_name)
        data = pd.concat([data.drop(labels=col_name, axis=1), categorical_data], axis=1)
        return data

    print(data.shape)
    print(len(col_name))
    if encoder is None:
        output_index = data.index
        # test_data可以传递测试数据，防止测试数据出现训练数据没有的分类变量
        data = data if test_data is None else pd.concat([data,test_data],axis=0)
        new_col_name = []
        mapping_dict = {}
        for col in col_name:
            # 创建分类变量取值到整数的映射，并建立离散化后的变量名
            all_value = set(data[col])
            mapping_dict[col] = dict(map(lambda x, y: (x, y), all_value, xrange(len(all_value))))
            data[col] = data[col].apply(lambda x: mapping_dict[col][x])
            new_col_name.extend(map(lambda x: col + '_' + str(x), xrange(len(all_value))))
        print(len(new_col_name))

        # 离散化分类变量
        enc = OneHotEncoder(categorical_features='all', sparse=False)
        categorical_data = pd.DataFrame(enc.fit_transform(data[col_name]),index=data.index,columns=new_col_name)
        # 分类变量替换为离散化变量
        final_data = concat_encoder(data,col_name,categorical_data,new_col_name)
        final_data = final_data.loc[output_index]
        return final_data, enc, mapping_dict, new_col_name

    else:
        assert test_data is None
        for col in col_name:
            data[col] = data[col].apply(lambda x: mapping_dict[col][x])
        categorical_data =  pd.DataFrame(encoder.transform(data[col_name]),index=data.index,columns=new_col_name)
        data = concat_encoder(data,col_name,categorical_data,new_col_name)
        return data

def feature_subgrouping(data, col_name):
    # 根据指定的工具变量确定变量的index
    col_index = map(lambda x: list(data.columns).index(x), col_name)
    col_index.append(data.shape[1])
    assert col_index == sorted(col_index)
    # 按顺序确定每个工具变量对应的特征
    feature_dict = {col_name[i]: data.columns[col_index[i]:col_index[i + 1]] for i in xrange(len(col_name))}
    return feature_dict

def chunk_dataframe_generator(data, feature_dict, category):
    assert category in feature_dict
    # 返回工具变量对应的特征矩阵
    return data.loc[:, feature_dict[category]]

def categorical_mean(df):
    for col in df.columns[1:]:
        df.loc[:,col] = df.loc[:,col].mean()
    return df

def new_categorical_df(data):
    categorical_columns = _identify_categorical_variable(data)
    feature_dict = feature_subgrouping(data, categorical_columns)
    final_df = pd.DataFrame(index=data.index)
    for category in categorical_columns:
        partial_df = chunk_dataframe_generator(data, feature_dict, category)
        partial_df = partial_df.groupby(category).apply(lambda x: categorical_mean(x))
        final_df = pd.concat([final_df, partial_df], axis=1)
    final_df.drop(categorical_columns,inplace=True,axis=1)
    print(final_df.shape)
    for col in final_df.columns:
        final_df.rename(columns={col: 'new_' + col}, inplace=True)
    return final_df

def categorical_processing(data, method='simple'):

    assert method in ['simple','complex']

    # simple选项下，每个工具变量离散化后去除冗余的一个变量
    if method == 'simple':
        discrete_col = _identify_categorical_variable(data)
        categorical_data = data.loc[:, discrete_col]
        categorical_data.columns = map(lambda x: x + '_new', categorical_data.columns)
        data = pd.concat([data, categorical_data], axis=1)
        data, enc, mapping_dict, new_col_name = categorical_encoding(data, categorical_data.columns)
        new_col_name = filter(lambda x: not re.match(r'.*\_0', x), new_col_name)
        data.drop(labels=filter(lambda x: re.match(r'.*\_0', x), data.columns), axis=1, inplace=True)

    # complex选项下，根据工具变量求均值，构造新的特征矩阵
    else:
        new_data = new_categorical_df(data)
        new_col_name = new_data.columns
        data = pd.concat([data, new_data], axis=1)

    return data, new_col_name


if __name__ == '__main__':
    '''
    category_n = 5
    train_data, train_score, test_data = cPickle.load(open('online_data.pkl'))
    data = pd.concat([train_data, test_data])

    categorical_columns = select_categorical(data, category_n)
    data, enc, min_ = categorical_encoding(data, categorical_columns)

    train_data = data.loc[train_data.index,:]
    test_data = data.loc[test_data.index,:]
    train_new_col_name = filter(lambda x:(train_data[x]==0).sum()!=499,train_data.columns)
    train_data = train_data.loc[:,train_new_col_name]
    test_data = test_data.loc[:,train_new_col_name]

    cPickle.dump((train_data, train_score, test_data), open('online_data_cate.pkl', 'w'))
    raise KeyboardInterrupt
    '''
    import cPickle
    data, score, _ = cPickle.load(open('online_data.pkl'))
    categorical_columns = ['TOOL','Tool','TOOL_ID','Tool (#1)','TOOL (#1)','TOOL (#2)','Tool (#2)','Tool (#3)','Tool (#4)','OPERATION_ID',
		        'Tool (#5)','TOOL (#3)']
    feature_dict = feature_subgrouping(data, categorical_columns)

    category = 'TOOL (#3)'
    df = chunk_dataframe_generator(data, feature_dict, category)
    pd.concat([df,score],axis=1).to_csv('explore/categorical_explore.csv')
    raise KeyboardInterrupt

    pd.concat([data.loc[:, category], df, df.isnull().sum(1)], axis=1).to_csv('data_explore.csv')

    na_count_df = pd.DataFrame(index=data.index)
    for category in feature_dict:
        df = chunk_dataframe_generator(data, feature_dict, category)
        na_count_df = pd.concat([na_count_df, data.loc[:, category], df.isnull().sum(1)], axis=1)
    na_count_df.to_csv('Categorical Variable.csv')
    '''
    train_data, train_score, test_data, test_score = cPickle.load(open('offline_data.pkl'))
    data = pd.concat([train_data, test_data], axis=0)
    data = new_categorical_df(data)
    train_data = data.loc[train_data.index, :]
    test_data = data.loc[test_data.index, :]

    cPickle.dump((train_data, test_data), open('offline_data_cf.pkl', 'w'))
    '''

