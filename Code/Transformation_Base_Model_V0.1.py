import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from fbprophet import Prophet


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
    #df.to_csv('df')
    print('exiting melt')


#Creating features for the model:

def train_features():

    global df_final
    print('entering train features')

    fun1=lambda x:1 if x==True else(0)

    df=pd.read_csv('df')
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

    start_time = time.time()
    df1=df['Sales']
    print('Sales time: {:.2f}s'.format(time.time() - start_time))

    start_time = time.time()
    df2 = df.drop('Sales',axis=1)
    print('Sales drop: {:.2f}s'.format(time.time() - start_time))

    start_time = time.time()
    df_final=pd.concat([df2,df1],axis=1)
    print('Concat: {:.2f}s'.format(time.time() - start_time))

    start_time = time.time()
    df_final.to_csv('df_final')
    print('To csv: {:.2f}s'.format(time.time() - start_time))

    print('exiting train features')


#Aggregating at the category level:

def cat_model():

    print('entering cat_agg')
    global df_final
    #df_cat = df_final.groupby(['cat_id', 'store_id', 'Date', 'dayofyear', 'weekofyear', 'monthstart', 'quarterstart', 'yearstart'])['Sales'].sum()
    #df_cat = df_dept.reset_index()


    df_cat = pd.read_csv('df_cat_pred')
    df_cat_pred = df_cat[['cat_id', 'store_id', 'Date', 'Sales']]
    df_cat_pred.columns = ['cat_id', 'store_id', 'ds', 'y']


    stores = {'CA_1': 1, 'CA_2': 2, 'CA_3': 3, 'CA_4': 4, 'TX_1': 5, 'TX_2': 6, 'TX_3': 7, 'WI_1': 8, 'WI_2': 9,'WI_3': 10}
    cat = {'FOODS': 1, 'HOBBIES': 2, 'HOUSEHOLD': 3}
    df_cat_pred['store_id'] = df_cat_pred['store_id'].map(stores)
    df_cat_pred['cat_id'] = df_cat_pred['cat_id'].map(cat)


    fb = Prophet(interval_width=0.95, daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    fb.add_country_holidays(country_name='US')
    fb.add_regressor('store_id')
    fb.add_regressor('cat_id')
    fb.fit(df_cat_pred)
    future = fb.make_future_dataframe(freq='D', periods=28, include_history=False)


    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    c = pd.Series([1, 2, 3])
    stores1 = pd.DataFrame({'store_id': s.repeat(28)}).reset_index()
    stores1 = pd.concat([stores1] * 3).reset_index().drop('index', axis=1)
    cat1 = pd.DataFrame({'cat_id': s.repeat(280)}).reset_index()


    final_df = pd.concat([future] * 30).reset_index().drop('index', axis=1)
    final_df['store_id'] = stores1['store_id']
    final_df['cat_id'] = cat1['cat_id']


    predict = fb.predict(final_df)
    y_pred_df = predict[['yhat']]


    cat_final = pd.concat([final_df, y_pred_df], axis=1)
    stores2 = {1: 'CA_1', 2: 'CA_2', 3: 'CA_3', 4: 'CA_4', 5: 'TX_1', 6: 'TX_2', 7: 'TX_3', 8: 'WI_1', 9: 'WI_2',10: 'WI_3'}
    cat2 = {1: 'FOODS', 2: 'HOBBIES', 3: 'HOUSEHOLD'}
    cat_final['store_id'] = cat_final['store_id'].map(stores2)
    cat_final['cat_id'] = cat_final['cat_id'].map(cat2)
    cat_final = cat_final.rename(columns={'ds': 'Date', 'yhat': 'Forecasted Sales'})
    cat_final = cat_final[['cat_id', 'store_id', 'Date', 'Forecasted Sales']]
    print(cat_final)


#Aggregating at the department level:

def dept_model():

    print('entering dept_agg')
    global df_dept,df_hobbies_1,df_hobbies_2,df_household_1,df_household_2,df_foods_1,df_foods_2,df_foods_3,df_foods_4
    #df_dept=df_final.groupby(['dept_id','store_id','Date','dayofyear','weekofyear','monthstart','quarterstart','yearstart'])['Sales'].sum()
    #df_dept=df_dept.reset_index()

    df_dept = pd.read_csv('df_dept_pred')
    df_dept = df_dept.rename(columns={'Sales':'Dept Sales'})
    df_dept_pred = df_dept[['dept_id', 'store_id', 'Date', 'Dept Sales']]
    df_dept_pred.columns = ['dept_id', 'store_id', 'ds', 'y']


    stores = {'CA_1': 1, 'CA_2': 2, 'CA_3': 3, 'CA_4': 4, 'TX_1': 5, 'TX_2': 6, 'TX_3': 7, 'WI_1': 8, 'WI_2': 9,'WI_3': 10}
    dept = {'FOODS_1': 1, 'FOODS_2': 2, 'FOODS_3': 3, 'HOUSEHOLD_1': 4, 'HOUSEHOLD_2': 5, 'HOBBIES_1': 6,'HOBBIES_2': 7}
    df_dept_pred['store_id'] = df_dept_pred['store_id'].map(stores)
    df_dept_pred['dept_id'] = df_dept_pred['dept_id'].map(dept)


    fb = Prophet(interval_width=0.95, daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    fb.add_country_holidays(country_name='US')
    fb.add_regressor('store_id')
    fb.add_regressor('dept_id')
    fb.fit(df_dept_pred)
    future = fb.make_future_dataframe(freq='D', periods=28, include_history=False)


    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    d = pd.Series([1, 2, 3, 4, 5, 6, 7])
    stores1 = pd.DataFrame({'store_id': s.repeat(28)}).reset_index()
    stores1 = pd.concat([stores1] * 7).reset_index().drop('index', axis=1)
    dept1 = pd.DataFrame({'dept_id': d.repeat(280)}).reset_index()


    final_df = pd.concat([future] * 70).reset_index().drop('index', axis=1)
    final_df['store_id'] = stores1['store_id']
    final_df['dept_id'] = dept1['dept_id']


    predict = fb.predict(final_df)
    y_pred_df = predict[['yhat']]


    final = pd.concat([final_df, y_pred_df], axis=1)
    stores2 = {1: 'CA_1', 2: 'CA_2', 3: 'CA_3', 4: 'CA_4', 5: 'TX_1', 6: 'TX_2', 7: 'TX_3', 8: 'WI_1', 9: 'WI_2',10: 'WI_3'}
    dept2 = {1: 'FOODS_1', 2: 'FOODS_2', 3: 'FOODS_3', 4: 'HOUSEHOLD_1', 5: 'HOUSEHOLD_2', 6: 'HOBBIES_1',7: 'HOBBIES_2'}
    final['store_id'] = final['store_id'].map(stores2)
    final['dept_id'] = final['dept_id'].map(dept2)
    final = final.rename(columns={'ds': 'Date', 'yhat': 'Forecasted Sales'})
    final = final[['dept_id', 'store_id', 'Date', 'Forecasted Sales']]
    print(final)


#Calculating the upc sales ratio per store:

def upc_prob():

    df_final=pd.read_csv('df_final')

    df_trial=df_final[['item_id','dept_id','store_id','Date','Sales']]
    df_dept_ratio=df_dept[['dept_id','store_id','Date','Dept Sales']]

    df_merge=df_trial.merge(df_dept,how='inner',on=['dept_id','store_id','Date'])
    print(df_merge.head(10))

    df_merge['Ratio'] = df_merge['Sales'] / df_merge['Dept Sales']
    df_merge['Ratio'] = df_merge['Ratio'].fillna(value=0)

    df_merge.to_pickle('merged')


#Calling the functions:

#process_train_cal()
#train_features()
#cat_model()
#dept_model()
#upc_prob()
