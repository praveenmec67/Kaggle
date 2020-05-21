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


#df_final = pd.read_pickle('df_final')
#df_cat=df_final.groupby(['cat_id','store_id','Date','dayofyear','weekofyear','monthstart','quarterstart','yearstart'])['Sales'].sum()
#df_cat=df_cat.reset_index()
#df_foods=df_cat[df_cat['cat_id']=='FOODS']


df_foods=pd.read_csv('df_foods')

df_foods=df_foods[['store_id','Date','Sales']]
df_foods.columns=['store_id','ds','y']
stores={'CA_1':1,'CA_2':2,'CA_3':3,'CA_4':4,'TX_1':5,'TX_2':6,'TX_3':7,'WI_1':8,'WI_2':9,'WI_3':10}
df_foods['store_id']=df_foods['store_id'].map(stores)

#df_foods_train=df_foods.iloc[1913:3796,:]
#df_foods_test=df_foods.iloc[3796:3826,:]
#y_test_df=df_foods_test[['y']].reset_index().drop('index',axis=1)
#print(y_test_df.head())
#y_test=df_foods_test['y']

#print(df_foods_train.head())
#print(df_foods_train.tail())
#print(df_foods_train.shape)
#print(df_foods_test.shape)



fb = Prophet(interval_width=0.95,yearly_seasonality=True,weekly_seasonality=True)
fb.add_country_holidays(country_name='US')
fb.add_regressor('store_id')
fb.fit(df_foods)
#fb.fit(df_foods_train)


future = fb.make_future_dataframe(freq='D',periods=28,include_history=False)
#future=df_foods_test['ds']
print(future)

a=pd.Series([1,2,3,4,5,6,7,8,9,10])
stores1=pd.DataFrame({'store_id':a.repeat(28)}).reset_index()


final_df=pd.concat([future]*10).reset_index().drop('index',axis=1)
#stores1=stores1.iloc[30:60,:].reset_index().drop('index',axis=1)
final_df['store_id']=stores1['store_id']
predict = fb.predict(final_df)
y_pred_df=predict[['yhat']]
#y_pred=predict['yhat']

final=pd.concat([final_df,y_pred_df],axis=1)
print(final.head())

#comp=pd.concat([final,y_test_df],axis=1)
#print(comp.head())

#res=mean_absolute_error(y_test,y_pred)
#mape=np.mean(np.abs((np.array(y_test) - np.array(y_pred)) / y_test)) * 100
#print(res)
#print(mape)


#fig=fb.plot(predict)
#plt.show()
#fig1=fb.plot_components(predict)
#plt.show()
