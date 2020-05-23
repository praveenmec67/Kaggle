import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter

#Setting the files location:
os.chdir('/Users/praveenthenraj/Desktop/PycharmProjects/Trial/Kaggle/Competitions')

df=pd.read_csv('df.csv')
df_g=df.groupby(['state_id','cat_id'])['Sales'].sum(ascending=False)
print(df_g)

x=np.arange(10)
def millions(x, pos):
    'The two args are the value and tick position'
    return '%1.1fM' % (x * 1e-6)


formatter = FuncFormatter(millions)
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(formatter)
df_g.plot.bar(stacked=True)
plt.xlabel('Category-State Combination')
plt.ylabel('Total Sales')
plt.title('Total Sales at Category-State Level')
plt.show()
