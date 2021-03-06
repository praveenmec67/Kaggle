import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
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
df2 = df2.drop(['predict_dates', 'date','events_type_1_1','events_type_2_1'], axis=1)
df2 = df2.reindex(columns=['store_id', 'snap_CA', 'snap_TX', 'snap_WI', 'events_name_1_1', 'events_name_2_1', 'dayofyear', 'dayofweek', 'weekofyear', 'weekend', 'monthstart', 'quarterstart','yearstart'])



#Into the Model:


#Hyperparameter Tuning:

n=[i for i in np.arange(100,300,100)]
l=[0.001,0.05,0.01,0.1]
#md=[i for i in range(4,8)]
#lam=[i for i in np.arange(0,1.1,0.1)]
#gam=[i for i in np.arange(0,1.1,0.1)]
param={'n_estimators':n,'learning_rate':l}
hyp=[]


start=time.time()
print('--------STARTING---------')

# Training file creation and model build:

for i in items:

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
        xgb1 = XGBRegressor(silent=True)

        start1=time.time()

        xgb = GridSearchCV(xgb1,param_grid=param,cv=3)
        xgb.fit(X, y)

        hyp.append(xgb.best_params_)
        y_pred=xgb.predict(X_pred)

        sub1=pd.concat([sub1,pd.DataFrame(y_pred)],axis=0,ignore_index=True)
        print(i +' : ' " %s seconds " % (time.time() - start1))
        print(xgb.best_params_)
        print()

stop = time.time()
print('---------COMPLETED-------')
print('Model run time  : '+' %s seconds ' % (time.time() - start))
hyp.to_csv('hyp.csv')
sub1.to_csv('prediction_file_raw.csv')

#Creating the output prediction file:

K = 56
elements = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
sub2['store_id'] = [ele for ele in elements for i in range(K)]
sub2['predict_dates'] = pd.concat([pd.DataFrame(pd.date_range(start='2016/04/25', end='2016/06/19'))] * 10,axis=0, ignore_index=True)
sub2=pd.concat([sub2]*3049).reset_index().drop('index',axis=1)

pred_sub=pd.concat([sub2,sub1],axis=1)
pred_sub.to_csv('pred_sub.csv')


#Submission file creation:

df1_sales_train=pd.read_csv('sales_train_validation.csv')
pred_file = pd.read_csv('pred_sub.csv')


id = df1_sales_train['id'].unique()
unique_values = [ele for ele in id for i in range(0,56)]
pred_file = pred_file.drop(['Unnamed: 0','store_id'],axis =1)
pred_file.columns=['Date','Sales']
pred_file['id'] = unique_values


pred_file = pred_file.pivot(index='id',columns='Date')


submission_01 = pd.DataFrame()
submission_02 = pd.DataFrame()
submission_01['id'] = pred_file.index
submission_02['id'] = pred_file.index
Temp_01 = pred_file['Sales'].iloc[:,0:28]
Temp_02 = pred_file['Sales'].iloc[:,28:]
submission_01 = pd.merge(left=submission_01,right=Temp_01,on=['id'],how='right')
submission_02 = pd.merge(left=submission_02,right = Temp_02 ,on=['id'],how='right')


Columns_Final_List = ['F{0}'.format(i) for i in range(0,29)]
submission_01.columns=Columns_Final_List
submission_02.columns=Columns_Final_List
Submission_File = pd.concat([submission_01,submission_02],axis=0,ignore_index=True)
Submission_File = Submission_File.rename(columns={'F0':'id'})
sample_submission = pd.read_csv('sample_submission.csv')
Submission_File['id'] = sample_submission['id']
cols = Submission_File.columns
Submission_File[cols[1:]] = Submission_File[cols[1:]].round(0).mask(Submission_File[cols[1:]]<=0,0)
Submission_File.to_csv('final_submission_xgb_base_tuned.csv',index=False)


stop = time.time()
print('Overall Run Time : '+' %s seconds ---' % (time.time() - start))

