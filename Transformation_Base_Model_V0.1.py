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

    start_time=time.time()
    df['dayofyear']=pd.DatetimeIndex(df['Date']).dayofyear
    print('dayofyear time: {:.2f}s'.format(time.time()-start_time))

    start_time = time.time()
    df['weekofyear']=pd.DatetimeIndex(df['Date']).weekofyear
    print('weekofyear time: {:.2f}s'.format(time.time() - start_time))

    start_time = time.time()
    df['monthstart']=pd.DatetimeIndex(df['Date']).is_month_start
    print('monthstart time: {:.2f}s'.format(time.time() - start_time))

    start_time = time.time()
    df['monthstart']=df['monthstart'].apply(fun1)
    print('monthstart apply time: {:.2f}s'.format(time.time() - start_time))

    start_time = time.time()
    df['quarterstart']=pd.DatetimeIndex(df['Date']).is_quarter_start
    print('quarterstart time: {:.2f}s'.format(time.time() - start_time))

    start_time = time.time()
    df['quarterstart']=df['quarterstart'].apply(fun1)
    print('quarterstart apply time: {:.2f}s'.format(time.time() - start_time))

    start_time = time.time()
    df['yearstart']=pd.DatetimeIndex(df['Date']).is_year_start
    print('yearstart time: {:.2f}s'.format(time.time() - start_time))

    start_time = time.time()
    df['yearstart']=df['yearstart'].apply(fun1)
    print('yearstart apply time: {:.2f}s'.format(time.time() - start_time))

    df1=df['Sales']

    df2 = df.drop('Sales',axis=1)

    df_final=pd.concat([df2,df1],axis=1)

    df_final.to_pickle('df_final')

    print('exiting train features')


#Aggregating at the category level:

def cat_agg():

    print('entering cat_agg')

    global df_foods,df_household,df_hobbies,df_final

    df_final = pd.read_pickle('df_final')

    df_cat=df_final.groupby(['cat_id','store_id','Date','dayofyear','weekofyear','monthstart','quarterstart','yearstart'])['Sales'].sum()

    df_cat=df_cat.reset_index()

    df_foods=df_cat[df_cat['cat_id']=='FOODS']

    df_household=df_cat[df_cat['cat_id']=='HOUSEHOLD']

    df_hobbies = df_cat[df_cat['cat_id'] == 'HOBBIES']

    print(df_hobbies.head(3))

    print('exiting cat_agg')


#Aggregating at the department level:

def dept_agg():

    print('entering dept_agg')

    global df_dept,df_hobbies_1,df_hobbies_2,df_household_1,df_household_2,df_foods_1,df_foods_2,df_foods_3,df_foods_4

    df_dept=df_final.groupby(['dept_id','store_id','Date','dayofyear','weekofyear','monthstart','quarterstart','yearstart'])['Sales'].sum()

    df_dept=df_dept.reset_index()

    df_dept=df_dept.rename(columns={'Sales':'Dept Sales'})

    df_hobbies_1=df_dept[df_dept['dept_id']=='HOBBIES_1']

    df_hobbies_1 = df_dept[df_dept['dept_id'] == 'HOBBIES_1']

    df_hobbies_2 = df_dept[df_dept['dept_id'] == 'HOBBIES_2']

    df_household_1 = df_dept[df_dept['dept_id'] == 'HOUSEHOLD_1']

    df_household_2 = df_dept[df_dept['dept_id'] == 'HOUSEHOLD_2']

    df_foods_1 = df_dept[df_dept['dept_id'] == 'FOODS_1']

    df_foods_2 = df_dept[df_dept['dept_id'] == 'FOODS_2']

    df_foods_3 = df_dept[df_dept['dept_id'] == 'FOODS_3']

    print(df_hobbies_1.head(3))
    print(df_hobbies_2.head(3))

    print('exiting dept_agg')

#Calculating the upc sales ratio per store:

def upc_prob():

  #  df_upc_prob=df_dept.groupby(['dept_id','store_id','Date'])['Sales'].sum()
  #  df_upc_prob=df_upc_prob.reset_index()
  #  df_upc_prob=df_upc_prob.rename(columns={'Sales':'Dept Sales'})
  #  df_upc_prob=df_upc_prob.ffill()
  #  df_upc_prob = df_dept.rename(columns={'Sales': 'Dept Sales'})

    df_trial=df_final[['item_id','dept_id','store_id','Date','Sales']]
    df_dept_ratio=df_dept[['dept_id','store_id','Date','Dept Sales']]

  # df_merge=df_trial.merge(df_upc_prob,how='inner',on=['dept_id','store_id','Date'])
    df_merge=df_trial.merge(df_dept,how='inner',on=['dept_id','store_id','Date'])
    print(df_merge.head(10))
    df_merge['Ratio'] = df_merge['Sales'] / df_merge['Dept Sales']


    print(df_merge.head(10))


#Calling the functions:

#process_train_cal()
#train_features()
cat_agg()
dept_agg()
upc_prob()
