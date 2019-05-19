# encoding:utf-8
# basic ensemble
import pandas as pd
import numpy as np
from collections import  Counter

char_cnn=pd.read_csv('out_char_cnn.csv',header=None)
char_bert=pd.read_csv('out_bert.csv',header=None)
char_gru_att=pd.read_csv('out_char_gru_att.csv',header=None)

out=pd.concat([char_cnn,char_bert.iloc[:,3:],char_gru_att.iloc[:,3:]],axis=1)
out= out.T.reset_index(drop=True).T
out.iloc[:,[5,8,11]].astype(np.float32)
# the maximum probability
out['index']=out.iloc[:,[5,8,11]].apply(lambda x: np.argmax(list(x)),axis=1)
out['class']=out.apply(lambda x: list(x)[3*(list(x)[12]+1)],axis=1)
# the most frequency
out['class_2']=out.iloc[:,[3,6,9]].apply(lambda x: max(dict(Counter(list(x))),key=dict(Counter(list(x))).get),axis=1)

out.to_csv("ensemble.csv",index=False,header=True,encoding='utf-8-sig')
