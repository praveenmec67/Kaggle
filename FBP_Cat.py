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


df_cat_pred=pd.read_csv('df_cat_pred')
print(df_cat_pred.tail())
df_cat_pred=df_cat_pred[['cat_id','store_id','Date','Sales']]
df_cat_pred.columns=['cat_id','store_id','ds','y']


stores={'CA_1':1,'CA_2':2,'CA_3':3,'CA_4':4,'TX_1':5,'TX_2':6,'TX_3':7,'WI_1':8,'WI_2':9,'WI_3':10}
cat={'FOODS':1,'HOBBIES':2,'HOUSEHOLD':3}
df_cat_pred['store_id']=df_cat_pred['store_id'].map(stores)
df_cat_pred['cat_id']=df_cat_pred['cat_id'].map(cat)



fb = Prophet(interval_width=0.95,daily_seasonality=True,weekly_seasonality=True,yearly_seasonality=True)
fb.add_country_holidays(country_name='US')
fb.add_regressor('store_id')
fb.add_regressor('cat_id')
fb.fit(df_cat_pred)
future = fb.make_future_dataframe(freq='D',periods=28,include_history=False)


s=pd.Series([1,2,3,4,5,6,7,8,9,10])
c=pd.Series([1,2,3])
stores1=pd.DataFrame({'store_id':s.repeat(28)}).reset_index()
stores1=pd.concat([stores1]*3).reset_index().drop('index',axis=1)
cat1=pd.DataFrame({'cat_id':s.repeat(280)}).reset_index()


final_df=pd.concat([future]*30).reset_index().drop('index',axis=1)
final_df['store_id']=stores1['store_id']
final_df['cat_id']=cat1['cat_id']


predict = fb.predict(final_df)
y_pred_df=predict[['yhat']]


final=pd.concat([final_df,y_pred_df],axis=1)
stores2={1:'CA_1',2:'CA_2',3:'CA_3',4:'CA_4',5:'TX_1',6:'TX_2',7:'TX_3',8:'WI_1',9:'WI_2',10:'WI_3'}
cat2={1:'FOODS',2:'HOBBIES',3:'HOUSEHOLD'}
final['store_id']=final['store_id'].map(stores2)
final['cat_id']=final['cat_id'].map(cat2)
final=final.rename(columns={'ds':'Date','yhat':'Forecasted Sales'})
final=final[['cat_id','store_id','Date','Forecasted Sales']]
print(final)
