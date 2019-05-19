import pandas as pd
import os
import re

def create_online(file):
    data = pd.read_excel(file)
    data['Mileage km']=data['Mileage km'].apply(lambda x: float(str(x).replace('km','')))
    data['Age year']=data['Age year'].apply(lambda x:float(str(x).replace('Year','')))
    data=data.loc[data['Fault location'].isin(['Engine','Car body','Door','Tire','Brake'])]
    data.to_csv('./raw/online_example.csv',index=False,header=True,encoding='utf-8-sig')


create_online('./raw/online_example_data.xlsx')