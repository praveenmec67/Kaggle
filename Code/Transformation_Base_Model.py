import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

#Setting the files location:
os.chdir('/Users/praveenthenraj/Desktop/PycharmProjects/Trial/Kaggle/Competitions')

#Processing calendar and training file:

def process_train_cal():

    global df
    df=pd.read_csv('sales_train_validation.csv')
    cal=pd.read_csv('calendar.csv')
    cal_d = cal['d'].tolist()
    cal_date = cal['date'].tolist()
    cal_dict = dict(zip(cal_d, cal_date))
    df1 =df.iloc[:,:6]
    df2 = df.iloc[:,6:].rename(columns=cal_dict)
    df=pd.concat([df1,df2],axis=1).drop('id',axis=1)
    df=pd.melt(id_vars=['item_id','dept_id','cat_id','store_id','state_id'],value_vars=df.iloc[:,5:],var_name='Date',value_name='Sales',frame=df)

#Creating features for the model:

def train_features():

    fun1=lambda x:1 if x==True else(0)

    df['dayofyear']=pd.DatetimeIndex(df['Date']).dayofyear
    df['weekofyear']=pd.DatetimeIndex(df['Date']).weekofyear
    df['monthstart']=pd.DatetimeIndex(df['Date']).is_month_start
    df['monthstart']=df['monthstart'].apply(fun1)
    df['quarterstart']=pd.DatetimeIndex(df['Date']).is_quarter_start
    df['quarterstart']=df['quarterstart'].apply(fun1)
    df['yearstart']=pd.DatetimeIndex(df['Date']).is_year_start
    df['yearstart']=df['yearstart'].apply(fun1)
    df1=df['Sales']
    df2 = df.drop('Sales',axis=1)
    df_final=pd.concat([df2,df1],axis=1)
    print(df_final.head())

process_train_cal()
train_features()
