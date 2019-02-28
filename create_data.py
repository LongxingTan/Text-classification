import pandas as pd
import os
import re

def preprocess(file_path,file_type='excel'):
    result=pd.DataFrame()
    for file in os.listdir(file_path):
        if file_type=='excel':
            data = pd.read_excel(os.path.join(file_path, file))
            data['text'] =  data['Description']
            data['part']=data['QFS_Tier1'].apply(lambda x: str(x).split(' - ')[0])
            data['type'] = data['QFS_Tier1'].apply(lambda x: str(x).split(' - ')[-1])
            result = result.append(data.loc[:, ['text', 'part', 'type']])
        elif file_type=='csv':
            data=pd.read_csv(os.path.join(file_path, file),sep=';')
            data['text']=data['Description']
            result=result.append(data.loc[:, ['text', 'QFS_Tier1', 'QFS_Tier3']])

    # basic cleaning
    result['text']=result['text'].apply(lambda x: re.sub(re.compile(r"\t| |\s"),'',str(x)))
    date_pattern=re.compile(r'(\d{4}-\d{1,2}-\d{1,2})')
    if_position=lambda x: [date.end() for date in date_pattern.finditer(x)]
    position=lambda x: max(if_position(x)) if if_position(x) else 0
    result['text']=result['text'].apply(lambda x: x[position(x):])
    #result['text']=result['text'].apply(lambda x: str(x).split('首次')[-1])
    #result['text']=result['text'].str.strip('：').str.strip('，')
    result['text']=result['text'].apply(lambda x: ','.join(re.split('[，：。,:.]', x)[1:7]))
    result['text_1']=result['text'].apply(lambda x: re.split('[,]',x)[0] if '购买' not in re.split('[,]',x)[0] else '')
    result['text_2'] = result['text'].apply(lambda x: '，'.join(re.split('[,]',x)[1:]))
    result['text']=result['text_1']+result['text_2']
    result['part']=result['part'].str.strip()
    result = result.loc[~result['part'].isin(['NA', 'Firecase', 'Delete', 'Na']), :]
    result = result.loc[~result['type'].isin(['NA', 'Delete', 'Na']), :]

    result['part'] = result['part'].apply(lambda x: str(x).replace('Brake Pad Pad','Brake Pad'))
    result['part'] = result['part'].apply(lambda x: str(x).replace('Air bag', 'Airbag'))
    result['part'] = result['part'].apply(lambda x: str(x).replace('CD/DVD driver', 'DVD/CD driver'))
    result['part'] = result['part'].apply(lambda x: str(x).replace('Engine poly-v-belt', 'Engine belt'))
    result['part'] = result['part'].apply(lambda x: str(x).replace('Key remote control', 'KEYLESS-GO'))
    result['part'] = result['part'].apply(lambda x: str(x).replace('Parkman', 'Parktronic'))
    #result['part'] = result['part'].apply(lambda x: str(x).replace('Multimedia', 'Multimedia/Speaker'))
    #result['part'] = result['part'].apply(lambda x: str(x).replace('Speaker', 'Multimedia/Speaker'))
    result['part'] = result['part'].apply(lambda x: str(x).replace('Steering Gear Gear', 'Steering gear'))
    result['part'] = result['part'].apply(lambda x: str(x).replace('Turn signal lamp', 'Turn signal light'))
    result['part'] = result['part'].apply(lambda x: str(x).replace('USB jerk', 'Usb'))
    result['part'] = result['part'].apply(lambda x: str(x).replace('USB port', 'Usb'))
    result['part'] = result['part'].apply(lambda x: str(x).replace('Interior old car', 'Interior'))
    result['part'] = result['part'].apply(lambda x: str(x).replace('Interior new car', 'Interior'))

    result['len']=result['text'].apply(lambda x: len(x))
    result=result.loc[result['len']>15,:]
    result['complaint']=result['part']+' '+result['type']
    result.drop(columns=['part','type','text_1','text_2','len'], inplace=True)
    result.drop_duplicates(inplace=True)

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
    preprocess('./data_raw',file_type='excel')
    print("Train, Dev and Test data created done :)")