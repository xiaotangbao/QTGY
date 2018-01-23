import pandas as pd
import cPickle
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
    tool_mark = re.compile(r'[A-Za-z]+_?[A-Za-z]+.*')
    categorical_columns = filter(lambda x: re.match(tool_mark, str(x)), df.columns)
    #return categorical_columns
    return ['TOOL', 'Tool', 'TOOL_ID', 'Tool (#1)', 'TOOL (#1)', 'TOOL (#2)', 'Tool (#2)', 'Tool (#3)', 'Tool (#4)',
     'OPERATION_ID','Tool (#5)', 'TOOL (#3)']

def categorical_encoding(data, col_name, encoder=None,mapping_dict=None,new_col_name=None, test_data=None):
    from sklearn.preprocessing import OneHotEncoder
    def concat_encoder(data,col_name,categorical_data,new_col_name):
        '''
        for col in col_name:
            col_ = filter(lambda x: x.startswith(col), new_col_name)
            data_front = data.loc[:, :col].drop(col, axis=1)
            data_back = data.loc[:, col:].drop(col, axis=1)
            data = pd.concat([data_front, categorical_data[col_], data_back], axis=1)
        '''
        assert categorical_data.shape[0] == data.shape[0]
        assert categorical_data.shape[1] == len(new_col_name)
        data = pd.concat([data.drop(labels=col_name, axis=1), categorical_data], axis=1)
        return data

    print(data.shape)
    print(len(col_name))
    if encoder is None:
        output_index = data.index
        data = data if test_data is None else pd.concat([data,test_data],axis=0)
        new_col_name = []
        mapping_dict = {}
        for col in col_name:
            all_value = set(data[col])
            mapping_dict[col] = dict(map(lambda x, y: (x, y), all_value, xrange(len(all_value))))
            data[col] = data[col].apply(lambda x: mapping_dict[col][x])
            new_col_name.extend(map(lambda x: col + '_' + str(x), xrange(len(all_value))))
        print(len(new_col_name))

        enc = OneHotEncoder(categorical_features='all', sparse=False)
        categorical_data = pd.DataFrame(enc.fit_transform(data[col_name]),index=data.index,columns=new_col_name)
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
    col_index = map(lambda x: list(data.columns).index(x), col_name)
    col_index.append(data.shape[1])
    feature_dict = {col_name[i]: data.columns[col_index[i]:col_index[i + 1]] for i in xrange(len(col_name))}
    return feature_dict

def chunk_dataframe_generator(data, feature_dict, category):
    assert category in feature_dict
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
    print(final_df.shape)
    final_df.drop(categorical_columns,inplace=True,axis=1)
    print(final_df.shape)
    for col in final_df.columns:
        final_df.rename(columns={col: 'new_' + col}, inplace=True)
    return final_df




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


    feature_dict = feature_subgrouping(data, categorical_columns)

    category = 'Tool (#1)'
    df = chunk_dataframe_generator(data, feature_dict, category)


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


