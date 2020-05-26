import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os




#Initial Settings:
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


os.chdir('/Users/praveenthenraj/Desktop/PycharmProjects/Trial/Kaggle/Competitions')


df=pd.read_csv('pred_sub.csv').drop('Unnamed: 0',axis=1)
print(df.shape)
df=df.sort_values(['store_id','predict_dates']).reset_index()
df2=pd.DataFrame()
df4=pd.DataFrame()
c = 2
i = 0
j = 85372
while(c<=21):

    if c%2==0:
        df1=df.iloc[i:j,:]
        df2=df2.append(df1)
        i=i+85372
        j=j+85372
        c=c+1
    else:
        df3=df.iloc[i:j,:]
        df4=df4.append(df3)
        i=i+85372
        j=j+85372
        c=c+1


print(df2.shape)
df2.to_csv('sub_0_28_day.csv')
print(df4.shape)
df4.to_csv('sub_28_56_day.csv')
