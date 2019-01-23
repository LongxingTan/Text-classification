import pandas as pd
import os
import re
import numpy as np

def preprocess(file_path):
    result=pd.DataFrame()
    for file in os.listdir(file_path):
        data=pd.read_excel(os.path.join(file_path,file))
        data['text']=data['Title']+data['Description']
        result=result.append(data.loc[:,['text','QFS_Tier1','QFS_Tier3']])

    # some basic cleaning
    result.dropna(inplace=True)
    result.drop(columns=['QFS_Tier3'],inplace=True)
    result.drop_duplicates(inplace=True)
    result['text']=result['text'].apply(lambda x: re.sub(re.compile(r"\t| |\s"),'',x))

    # split into train, dev and test
    indices=np.random.permutation(range(result.shape[0]))
    train_indices=int(0.8*len(indices))
    dev_indices=int(0.9*len(indices))
    test_indeces=int(len(indices))
    train=result.loc[:train_indices,:]
    train.to_csv("./data/train.csv",index=False,header=False,encoding='utf-8-sig')
    dev=result.loc[train_indices:dev_indices,:]
    dev.to_csv("./data/dev.csv",index=False,header=False,encoding='utf-8-sig')
    test=result.loc[dev_indices:test_indeces,:]
    test.to_csv("./data/test.csv", index=False,header=False,encoding='utf-8-sig')

if __name__=='__main__':
    preprocess('./data_raw')
    print("Train Dev and Test data created done :)")
