import pandas as pd
import numpy as np
import patsy
import matplotlib.pyplot as plt

columns=pd.date_range(pd.Timestamp.now(),freq="D",periods=5).round(freq='D')

data = pd.DataFrame({ \
'x0':columns, \
'y':[-1.5,0.,3.6,1.3,3.]})

data1 = data
data1['x0'] = data1['x0']-data1['x0'].min()

# In [90]: data1
# Out[90]:
#       x0    y
# 0 0 days -1.5
# 1 1 days  0.0
# 2 2 days  3.6
# 3 3 days  1.3
# 4 4 days  3.0

# The Series class has a pandas.Series.dt accessor object with several useful
# datetime attributes, including dt.days. Access this attribute via:
#  timedelta_series.dt.days

data1['x0'] = data1['x0'].dt.days

y,X = patsy.dmatrices('y ~ x0',data1)

coef, resid, _, _ = np.linalg.lstsq(X,y)

y_predict = coef[0] + coef[1]*data1['x0']

plt.scatter(columns,y,alpha=0.3)
plt.xticks(rotation=90)
plt.plot(columns,y_predict, linewidth = 3)

plt.show()

# næste skridt er at udnytte patsy "category" funtionalitet når man giver en
# string kolonne istedet for numerisk kolonne (skal huske at håndtere intercept)
