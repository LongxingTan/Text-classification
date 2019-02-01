import pandas as pd
import os
import re
import random
import numpy as np

def preprocess(file_path,file_type='excel'):
    result=pd.DataFrame()
    for file in os.listdir(file_path):
        if file_type=='excle':
            data = pd.read_excel(os.path.join(file_path, file))
            data['text'] = data['Title'] + data['Description']
            result = result.append(data.loc[:, ['text', 'QFS_Tier1', 'QFS_Tier3']])
        elif file_type=='csv':
            data=pd.read_csv(os.path.join(file_path, file),sep=';')
            data['text']=data['Description'].apply(lambda x: str(x).split('投诉')[-1])
            result=result.append(data.loc[:, ['text', 'QFS_Tier1', 'QFS_Tier3']])


    # some basic cleaning
    result.dropna(inplace=True)
    result.drop(columns=['QFS_Tier3'],inplace=True)
    result.drop_duplicates(inplace=True)
    result['text']=result['text'].apply(lambda x: re.sub(re.compile(r"\t| |\s"),'',x))

    # split into train, dev and test
    result=result.sample(frac=1).reset_index(drop=True)
    split1=int(0.9*len(result))
    split2=int(0.95*len(result))

    train=result.iloc[:split1,:]
    train.to_csv("./data/train.csv",index=False,header=False,encoding='utf-8-sig')
    dev=result.iloc[split1:split2,:]
    dev.to_csv("./data/dev.csv",index=False,header=False,encoding='utf-8-sig')
    test=result.iloc[split2:,:]
    test.to_csv("./data/test.csv", index=False,header=False,encoding='utf-8-sig')

if __name__=='__main__':
    preprocess('./data_raw',file_type='csv')
    print("Train Dev and Test data created done :)")

