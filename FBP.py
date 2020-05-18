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


df_final = pd.read_pickle('df_final')
df_cat=df_final.groupby(['cat_id','store_id','Date','dayofyear','weekofyear','monthstart','quarterstart','yearstart'])['Sales'].sum()
df_cat=df_cat.reset_index()
df_foods=df_cat[df_cat['cat_id']=='FOODS']


df_foods=df_foods[['Date','Sales']]
df_foods.columns=['ds','y']
df_foods_train=df_foods.iloc[0:1000,:]
df_foods_test=df_foods.iloc[1000:1913,:]

fb = Prophet(interval_width=0.95,yearly_seasonality=True,weekly_seasonality=True)
fb.add_country_holidays(country_name='US')
fb.fit(df_foods_train)
future = fb.make_future_dataframe(freq='D', periods=913,include_history=False)
predict = fb.predict(future)


res=mean_absolute_error(df_foods_test['y'],predict['yhat'])
print(res)

fig=fb.plot(predict)
plt.show()
fig1=fb.plot_components(predict)
plt.show()