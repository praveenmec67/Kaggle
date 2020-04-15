import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error
import os

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)


#Setting the files location:
os.chdir('/Users/praveenthenraj/Desktop/PycharmProjects/Trial/Kaggle/Competitions')


#Processing calendar data:

def process_cal(state):
    global cal
    global name
    cal=pd.read_csv('calendar.csv')
    cal=pd.DataFrame(cal.iloc[0:1913,:])

    #Processing the date field to get the monthstart,day of year,week of year,year start columns:

    fun1=lambda x:1 if(x==True) else(0)

    cal['monthstart']= pd.DatetimeIndex(cal['date']).is_month_start
    cal['monthstart']=cal['monthstart'].apply(fun1)
    cal['dayofyear']= pd.DatetimeIndex(cal['date']).dayofyear
    cal['weekofyear']= pd.DatetimeIndex(cal['date']).weekofyear
    cal['yearstart']= pd.DatetimeIndex(cal['date']).is_year_start
    cal['yearstart']=cal['yearstart'].apply(fun1)


    #Processing categorical data to get dummies and concatenating with the original cal data to get the state-wise cal data:

    cal_ms=pd.get_dummies(cal['monthstart'],prefix='Monthstart')
    cal_ys=pd.get_dummies(cal['yearstart'],prefix='Yearstart')
    if state=='CA':
        ca=pd.get_dummies(cal.iloc[:,11],prefix_sep='_',prefix='Snap')
    elif state== 'TX':
        tx=pd.get_dummies(cal.iloc[:,12],prefix_sep='_',prefix='Snap')
    else:
        wi=pd.get_dummies(cal.iloc[:,13],prefix_sep='_',prefix='Snap')

    cal['event_name_1']=cal['event_name_1'].fillna(value='No Event1')
    cal['event_name_2']=cal['event_name_2'].fillna(value='No Event2')
    cal_en1=pd.get_dummies(cal['event_name_1'],prefix_sep='_',prefix=None)
    cal_en2=pd.get_dummies(cal['event_name_2'],prefix_sep='_',prefix=None)


    cal=cal.drop(['monthstart','yearstart','snap_CA','snap_TX','snap_WI','event_name_1','event_type_1','event_name_2','event_type_2'],axis=1)
    cal=pd.concat([cal,cal_ms,cal_ys,cal_en1,cal_en2],axis=1)

    name='cal_'+state
    if state=='CA':
        name = pd.concat([cal,ca],axis=1)
        name = name.set_index('date')

    elif state=='TX':
        name = pd.concat([cal,tx], axis=1)
        name = name.set_index('date')
    else:
        name = pd.concat([cal,wi], axis=1)
        name = name.set_index('date')


def process_train(state):

    global train_name
    global train_name1
    global final

    #Processing train file:
    df = pd.read_csv('sales_train_validation.csv')
    cal_d=cal['d'].tolist()
    cal_date=cal['date'].tolist()
    cal_dict=dict(zip(cal_d,cal_date))
    df=df.rename(columns=cal_dict)

    #Filtering state level sales data and aggregating the total sales in total column in train data:
    train_name='df_'+state
    train_name=df[df['state_id']==state]

    train_name1='df_total'+state
    df_train_name1=pd.DataFrame(train_name.iloc[:,6:].sum(axis=0)).rename(columns={0:'total'})


    #Concatenating the total sales data state-wise with the already processed state-wise calendar data:

    final='cal_df_'+state
    final=pd.concat([name,df_train_name1],axis=1).reset_index()


    def model():

    # Building a Model:

        X=final.drop(['date','wm_yr_wk','weekday','year','d','total'],axis=1).values
        y=final['total'].values

        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=20)

        lin=LinearRegression()
        lin.fit(X_train,y_train)

        cal_df_CA1=final.drop(['date','wm_yr_wk','weekday','year','d','total'],axis=1)
        col=list(cal_df_CA1.columns)
        coeff=list(lin.coef_)
        fin=dict(zip(col,coeff))

        print('Correlation-coefficient for state : '+state)
        print()
        for i in fin:
            if fin.get(i)>=0:
                print(i +' : '+ str(fin.get(i)))
        else:
            pass
        print()

    model()


#Calling the functions for each state:
process_cal('CA')
process_train('CA')
process_cal('TX')
process_train('TX')
process_cal('WI')
process_train('WI')

