import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

os.chdir('/Users/praveenthenraj/Desktop/PycharmProjects/Trial/Kaggle/Competitions')

#General Processing:
df_sales_train=pd.read_csv('sales_train_validation.csv')
cal=pd.read_csv('calendar.csv')
cal_train=pd.read_csv('cal_train')
print(cal_train.columns)
cal_train=cal_train.drop(['Unnamed: 0', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year','d'],axis=1)
print(type(cal_train))
cal_d = cal['d'].tolist()
cal_date = cal['date'].tolist()
cal_dict = dict(zip(cal_d, cal_date))
df1_sales_train=df_sales_train.iloc[:,:6]
df2_sales_train = df_sales_train.iloc[:,6:].rename(columns=cal_dict)
df_master=pd.concat([df1_sales_train,df2_sales_train],axis=1).drop('id',axis=1)

items=df_master['item_id'].unique()

state=df_master['state_id'].unique()
stores=df_master['store_id'].unique()
stores1={'CA_1':1,'CA_2':2,'CA_3':3,'CA_4':4,'TX_1':5,'TX_2':6,'TX_3':7,'WI_1':8,'WI_2':9,'WI_3':10}
state1={'CA':1,'TX':2,'WI':3}
fun=lambda x: 1 if (x == 5) | (x == 6) else (0)
fun1=lambda x:1 if x==True else(0)
df_comp=pd.DataFrame()
result=[]
#for i in range(1,df.shape[0]-30487):
j=1

start=time.time()
print(start)


for i in items:
     if j<=50:
        print('entering for')
        print(i)
        df1=df_master.loc[df_master['item_id']==i]
        df1=pd.melt(id_vars=['store_id'],value_vars=df_master.iloc[:,5:],var_name='Date',value_name='Sales',frame=df1)
        df1=df1.sort_values(['store_id','Date'])
        df1=pd.merge(df1, cal_train, how='left', left_on=['Date'], right_on=['date'])
        df1['dayofyear'] = pd.DatetimeIndex(df1['Date']).dayofyear
        df1['dayofweek'] = pd.DatetimeIndex(df1['Date']).dayofweek
        df1['weekofyear'] = pd.DatetimeIndex(df1['Date']).weekofyear
        df1['weekend'] = df1['dayofweek'].apply(fun)
        df1['monthstart'] = pd.DatetimeIndex(df1['Date']).is_month_start
        df1['monthstart'] = df1['monthstart'].apply(fun1)
        df1['quarterstart'] = pd.DatetimeIndex(df1['Date']).is_quarter_start
        df1['quarterstart'] = df1['quarterstart'].apply(fun1)
        df1['yearstart'] = pd.DatetimeIndex(df1['Date']).is_year_start
        df1['yearstart'] = df1['yearstart'].apply(fun1)
        df1['store_id']=df1['store_id'].map(stores1)
        df1 = df1.drop(['date','Date'], axis=1)
        df1_Sales = df1['Sales']
        df1 = df1.drop('Sales', axis=1)
        df1 = pd.concat([df1, df1_Sales], axis=1)



        X=df1.drop('Sales',axis=1).values
        y=df1['Sales'].values
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
        xgb=XGBRegressor()
        xgb.fit(X_train,y_train)
        y_pred=xgb.predict(X_test)
        df_comp['y_test']=y_test
        df_comp['y_pred']=y_pred
        df_comp['y_pred']=df_comp['y_pred'].round()
        rmse=np.sqrt(mean_squared_error(y_test,np.round(y_pred)))
        result.append(rmse)
        j=j+1
     else:
        break


stop=time.time()
print("--- %s seconds ---" % (time.time() - start))
print(stop)
print(j)
print(np.mean(result))



