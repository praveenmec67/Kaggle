import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import time


#Initial Settings:
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

os.chdir('/Users/praveenthenraj/Desktop/PycharmProjects/Trial/Kaggle/Competitions')


# General Processing:

df_sales_train = pd.read_csv('sales_train_validation.csv')
cal = pd.read_csv('calendar.csv')
cal_train = pd.read_csv('cal_train')
cal_pred = pd.read_csv('cal_pred')
cal_train = cal_train.drop(['Unnamed: 0', 'wm_yr_wk', 'events_type_1_1','events_type_2_1','weekday', 'wday', 'month', 'year', 'd'], axis=1)
cal_d = cal['d'].tolist()
cal_date = cal['date'].tolist()
cal_dict = dict(zip(cal_d, cal_date))
df1_sales_train = df_sales_train.iloc[:, :6]
df2_sales_train = df_sales_train.iloc[:, 6:].rename(columns=cal_dict)
df_master = pd.concat([df1_sales_train, df2_sales_train], axis=1).drop('id', axis=1)



#Initializing and Mapping:
items = df_master['item_id'].unique()
state = df_master['state_id'].unique()
stores = df_master['store_id'].unique()
df_comp = pd.DataFrame()
predict_df = pd.DataFrame()
y_pred_series=pd.Series()
sub1=pd.DataFrame()
sub2=pd.DataFrame()
result = []
stores1 = {'CA_1': 1, 'CA_2': 2, 'CA_3': 3, 'CA_4': 4, 'TX_1': 5, 'TX_2': 6, 'TX_3': 7, 'WI_1': 8, 'WI_2': 9,'WI_3': 10}
state1 = {'CA': 1, 'TX': 2, 'WI': 3}
fun = lambda x: 1 if (x==4)| (x == 5) | (x == 6) else (0)
fun1 = lambda x: 1 if x == True else (0)




# Creation of prediction file:

K = 56
elements = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
predict_df['store_id'] = [ele for ele in elements for i in range(K)]
predict_df['predict_dates'] = pd.concat([pd.DataFrame(pd.date_range(start='2016/04/25', end='2016/06/19'))] * 10,axis=0, ignore_index=True)
predict_df['dayofyear'] = pd.DatetimeIndex(predict_df['predict_dates']).dayofyear
predict_df['dayofweek'] = pd.DatetimeIndex(predict_df['predict_dates']).dayofweek
predict_df['weekofyear'] = pd.DatetimeIndex(predict_df['predict_dates']).weekofyear
predict_df['weekend'] = predict_df['dayofweek'].apply(fun)
predict_df['monthstart'] = pd.DatetimeIndex(predict_df['predict_dates']).is_month_start
predict_df['monthstart'] = predict_df['monthstart'].apply(fun1)
predict_df['quarterstart'] = pd.DatetimeIndex(predict_df['predict_dates']).is_quarter_start
predict_df['quarterstart'] = predict_df['quarterstart'].apply(fun1)
predict_df['yearstart'] = pd.DatetimeIndex(predict_df['predict_dates']).is_year_start
predict_df['yearstart'] = predict_df['yearstart'].apply(fun1)
cal_pred['date'] = pd.to_datetime(cal_pred['date'])
df2 = pd.merge(cal_pred, predict_df, how='left', left_on=['date'], right_on=['predict_dates'])
df2 = df2.sort_values(['store_id', 'date'])
print(df2.head(60))
df2 = df2.drop(['predict_dates', 'date','events_type_1_1','events_type_2_1'], axis=1)
df2 = df2.reindex(columns=['store_id', 'snap_CA', 'snap_TX', 'snap_WI', 'events_name_1_1', 'events_name_2_1', 'dayofyear', 'dayofweek', 'weekofyear', 'weekend', 'monthstart', 'quarterstart','yearstart'])



#Into the Model:
start = time.time()
j = 1

# Training file creation and model build:
for i in items:

        print(i)
        df1 = df_master.loc[df_master['item_id'] == i]
        df1 = pd.melt(id_vars=['store_id'], value_vars=df_master.iloc[:, 5:], var_name='Date', value_name='Sales',frame=df1)
        df1 = df1.sort_values(['store_id', 'Date'])
        df1 = pd.merge(df1, cal_train, how='left', left_on=['Date'], right_on=['date'])
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
        df1['store_id'] = df1['store_id'].map(stores1)
        df1 = df1.drop(['date', 'Date'], axis=1)
        df1_Sales = df1['Sales']
        df1 = df1.drop('Sales', axis=1)
        df1 = pd.concat([df1, df1_Sales], axis=1)


        X = df1.drop('Sales', axis=1).values
        y = df1['Sales'].values
        X_pred = df2.values
        xgb = XGBRegressor()
        xgb.fit(X, y)
        y_pred = xgb.predict(X_pred)
        sub1=pd.concat([sub1,pd.DataFrame(y_pred)],axis=0,ignore_index=True)
        j = j + 1


stop = time.time()
print("--- %s seconds ---" % (time.time() - start))



#Creating the output prediction file:

K = 56
elements = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
sub2['store_id'] = [ele for ele in elements for i in range(K)]
sub2['predict_dates'] = pd.concat([pd.DataFrame(pd.date_range(start='2016/04/25', end='2016/06/19'))] * 10,axis=0, ignore_index=True)
sub2=pd.concat([sub2]*3049).reset_index().drop('index',axis=1)

pred_sub=pd.concat([sub2,sub1],axis=1)
print(pred_sub.head())
print(pred_sub.iloc[:560,:])
print(pred_sub.tail())
pred_sub.to_csv('pred_sub.csv')

#Transformation:

df=pd.read_csv('pred_sub.csv').drop('Unnamed: 0',axis=1)
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


stop = time.time()
print("--- %s seconds ---" % (time.time() - start))

