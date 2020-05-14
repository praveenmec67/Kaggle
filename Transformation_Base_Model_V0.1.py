import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time



pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

#Setting the files location:
os.chdir('/Users/praveenthenraj/Desktop/PycharmProjects/Trial/Kaggle/Competitions')

#Processing calendar and training file:

def process_train_cal():

    global df

    df=pd.read_csv('sales_train_validation.csv')

    print(df.dept_id.unique())
    print(df.cat_id.unique())

    cal=pd.read_csv('calendar.csv')

    cal_d = cal['d'].tolist()

    cal_date = cal['date'].tolist()

    cal_dict = dict(zip(cal_d, cal_date))

    df1 =df.iloc[:,:6]

    df2 = df.iloc[:,6:].rename(columns=cal_dict)

    df=pd.concat([df1,df2],axis=1).drop('id',axis=1)

    print('entering melt')
    df=pd.melt(id_vars=['item_id','dept_id','cat_id','store_id','state_id'],value_vars=df.iloc[:,5:],var_name='Date',value_name='Sales',frame=df)
    print('exiting melt')

#Creating features for the model:

def train_features():

    global df_final

    print('entering train features')

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

    print('exiting train features')

#Aggregating at the category level:

def cat_agg():

    print('entering cat_agg')

    global df_foods,df_household,df_hobbies

    df_cat=df_final.groupby(['cat_id','store_id','Date','dayofyear','weekofyear','monthstart','quarterstart','yearstart'])['Sales'].sum()

    df_cat=df_cat.reset_index()

    df_foods=df_cat[df_cat['cat_id']=='FOODS']

    df_household=df_cat[df_cat['cat_id']=='HOUSEHOLD']

    df_hobbies = df_cat[df_cat['cat_id'] == 'HOBBIES']

    print(df_foods.head())

    print('exiting cat_agg')

#Aggregating at the department level:

def dept_agg():

    print('entering dept_agg')

    global df_hobbies_1,df_hobbies_2,df_household_1,df_household_2,df_foods_1,df_foods_2,df_foods_3,df_foods_4

    df_dept=df_final.groupby(['dept_id','store_id','Date','dayofyear','weekofyear','monthstart','quarterstart','yearstart'])['Sales'].sum()

    df_dept=df_dept.reset_index()

    df_hobbies_1=df_dept[df_dept['dept_id']=='HOBBIES_1']

    df_hobbies_1 = df_dept[df_dept['dept_id'] == 'HOBBIES_1']

    df_hobbies_2 = df_dept[df_dept['dept_id'] == 'HOBBIES_2']

    df_household_1 = df_dept[df_dept['dept_id'] == 'HOUSEHOLD_1']

    df_household_2 = df_dept[df_dept['dept_id'] == 'HOUSEHOLD_2']

    df_foods_1 = df_dept[df_dept['dept_id'] == 'FOODS_1']

    df_foods_2 = df_dept[df_dept['dept_id'] == 'FOODS_2']

    df_foods_3 = df_dept[df_dept['dept_id'] == 'FOODS_3']


    print(df_foods_1.head())
    print(df_foods_2.head())
    print(df_foods_3.head())


    print('exiting dept_agg')


#Calling the functions:

process_train_cal()
train_features()
cat_agg()
dept_agg()
