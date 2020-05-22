import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)

#Setting the files location:
os.chdir('/Users/praveenthenraj/Desktop/PycharmProjects/Trial/Kaggle/Competitions')


df_dept_pred=pd.read_csv('df_dept_pred')
df_dept_pred=df_dept_pred[['dept_id','store_id','Date','Dept Sales']]
df_dept_pred.columns=['dept_id','store_id','ds','y']


stores={'CA_1':1,'CA_2':2,'CA_3':3,'CA_4':4,'TX_1':5,'TX_2':6,'TX_3':7,'WI_1':8,'WI_2':9,'WI_3':10}
dept={'FOODS_1':1,'FOODS_2':2,'FOODS_3':3,'HOUSEHOLD_1':4,'HOUSEHOLD_2':5,'HOBBIES_1':6,'HOBBIES_2':7}
df_dept_pred['store_id']=df_dept_pred['store_id'].map(stores)
df_dept_pred['dept_id']=df_dept_pred['dept_id'].map(dept)


fb = Prophet(interval_width=0.95,daily_seasonality=True,weekly_seasonality=True,yearly_seasonality=True)
fb.add_country_holidays(country_name='US')
fb.add_regressor('store_id')
fb.add_regressor('dept_id')
fb.fit(df_dept_pred)
future = fb.make_future_dataframe(freq='D',periods=28,include_history=False)


s=pd.Series([1,2,3,4,5,6,7,8,9,10])
d=pd.Series([1,2,3,4,5,6,7])
stores1=pd.DataFrame({'store_id':s.repeat(56)}).reset_index()
stores1=pd.concat([stores1]*7).reset_index().drop('index',axis=1)
dept1=pd.DataFrame({'dept_id':d.repeat(560)}).reset_index()


final_df=pd.concat([future]*70).reset_index().drop('index',axis=1)
final_df['store_id']=stores1['store_id']
final_df['dept_id']=dept1['dept_id']


predict = fb.predict(final_df)
y_pred_df=predict[['yhat']]

final=pd.concat([final_df,y_pred_df],axis=1)
stores2={1:'CA_1',2:'CA_2',3:'CA_3',4:'CA_4',5:'TX_1',6:'TX_2',7:'TX_3',8:'WI_1',9:'WI_2',10:'WI_3'}
dept2={1:'FOODS_1',2:'FOODS_2',3:'FOODS_3',4:'HOUSEHOLD_1',5:'HOUSEHOLD_2',6:'HOBBIES_1',7:'HOBBIES_2'}
final['store_id']=final['store_id'].map(stores2)
final['dept_id']=final['dept_id'].map(dept2)
final=final.rename(columns={'ds':'Date','yhat':'Forecasted Sales'})
final=final[['dept_id','store_id','Date','Forecasted Sales']]
print(final)
