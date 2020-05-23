import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#Setting the files location:
os.chdir('/Users/praveenthenraj/Desktop/PycharmProjects/Trial/Kaggle/Competitions')

df=pd.read_csv('df.csv')
print(df.head())
print('entering')
df_g=df.groupby(['store_id','item_id'])['Sales'].sum()
df_g=df_g.reset_index()
values=['store_id','item_id','Sales']
df_g.to_csv('df_upc_store_grpby',columns=values)


df_g=pd.read_csv('df_upc_store_grpby').drop('Unnamed: 0',axis=1)
fig, axs = plt.subplots(5,2, figsize=(15, 6))

def func(a,status):
    j=0;k=0;l=0
    if status=='top':
        bool=False
        title_text='Top'
        color='g'
        color1='b'
        x=0.9
        y='0.7'
    else:
        bool=True
        title_text='Low'
        color='r'
        color1='b'
        x=0.1
        y=0.8
    for i in a:
            name='df_'+i
            name=df_g[df_g['store_id']==i]
            name=name.sort_values('Sales',ascending=bool,inplace=False).head()
            axs[j,k].bar(name['item_id'],name['Sales'],color=color)
            axs[j,k].set_title(i,x=x,y=y,fontsize=8,color=color1)
            axs[j,k].tick_params(axis='both', which='major',labelsize=6)
            l=l+1
            if l==2:
                j=j+1
                k=0
                l=0
            else:
                k=k+1

    plt.suptitle(title_text+' Sold Products in every Store',fontsize=18)
    plt.show()



func(['CA_1','CA_2','CA_3','CA_4','TX_1','TX_2','TX_3','WI_1','WI_2','WI_3'],'low')

