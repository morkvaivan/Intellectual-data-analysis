from pandas import read_csv, DataFrame
import numpy as np
import statsmodels.api as sm
from statsmodels.iolib.table import SimpleTable
from sklearn.metrics import r2_score
import ml_metrics as metrics
import matplotlib
from statsmodels.graphics import utils
from statsmodels.tsa.stattools import acf, pacf
from matplotlib import *
import sys
from pylab import *
from statsmodels.tsa.arima_model import ARIMA

dataset = read_csv('data_moving.csv',';', index_col=['date_oper'], parse_dates=['date_oper'], dayfirst=True)
dataset.head()

otg = dataset.Otgruzka
otg.head()

otg.plot(figsize=(12,6))

otg = otg.resample('W', how='mean')
otg.plot(figsize=(12,6))

itog = otg.describe()
otg.hist()
itog

print ('V = %f' % (itog['std']/itog['mean']))

row =  [u'JB', u'p-value', u'skew', u'kurtosis']
jb_test = sm.stats.stattools.jarque_bera(otg)
a = np.vstack([jb_test])
itog = SimpleTable(a, row)
print (itog)

test = sm.tsa.adfuller(otg)
print ('adf: ', test[0]) 
print ('p-value: ', test[1])
print ('Critical values: ', test[4])

otg1diff = otg.diff(periods=1).dropna()

test = sm.tsa.adfuller(otg1diff)
print ('adf: ', test[0])
print ('p-value: ', test[1])
print ('Critical values: ', test[4])
if test[0]> test[4]['5%']: 
    print ('there are single roots, the series is not stationary')
else:
    print ('there are no unit roots, the row is stationary')

otg1diff.plot(figsize=(12,6))

fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(otg1diff.values.squeeze(), lags=25, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(otg1diff, lags=25, ax=ax2)

src_data_model = otg[0:]
src_data_model
model = sm.tsa.ARIMA(src_data_model, order=(1,0,1), freq='W').fit(full_output=True, disp=0)

print (model.summary())

q_test = sm.tsa.stattools.acf(model.resid, qstat=True) 

print (DataFrame({'Q-stat':q_test[1], 'p-value':q_test[2]}))

pred = model.predict(1,150)
trn = otg[1:150]

r2 = r2_score(trn, pred[1:150])
print ('R^2: %1.2f' % r2)

metrics.rmse(trn,pred[1:150])

metrics.mae(trn,pred[1:150])

otg.plot(figsize=(12,6))
pred.plot(style='r--')

