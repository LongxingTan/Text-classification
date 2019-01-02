#

import pandas as pd
import os

def preprocess(file_path):
    result=pd.DataFrame()
    for file in os.listdir(file_path):
        data=pd.read_excel(os.path.join(file_path,file))
        data['text']=data['Title']+data['Description']
        result=result.append(data.loc[:,['text','QFS_Tier1','QFS_Tier3']])
    print(result.head(5))



preprocess('./raw_data')
