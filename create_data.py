import pandas as pd
import os
import re


def preprocess(file_path,file_type='excel'):
    result=pd.DataFrame()
    for file in os.listdir(file_path):
        if file_type=='excel':
            data = pd.read_excel(os.path.join(file_path, file))
            data['text'] = data['Title'] + data['Description']
            result = result.append(data.loc[:, ['text', 'QFS_Tier1', 'QFS_Tier3']])
        elif file_type=='csv':
            data=pd.read_csv(os.path.join(file_path, file),sep=';')
            data['text']=data['Description']
            result=result.append(data.loc[:, ['text', 'QFS_Tier1', 'QFS_Tier3']])

    # basic cleaning
    result.drop(columns=['QFS_Tier3'],inplace=True)
    result.drop_duplicates(inplace=True)
    result.dropna(inplace=True)
    result['text']=result['text'].apply(lambda x: re.sub(re.compile(r"\t| |\s"),'',x))
    result['text']=result['text'].apply(lambda x: str(x).split('首次')[-1])
    result['text']=result['text'].str.strip('：').str.strip('，')
    #result['text']=result['text'].apply(lambda x: ''.join(re.split('[，：。]', x)[2:4]))
    result['QFS_Tier1']=result['QFS_Tier1'].str.strip()

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
    print("Train, Dev and Test data created done :)")

