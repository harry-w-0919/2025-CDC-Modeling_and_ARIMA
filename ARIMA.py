import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df = pd.read_csv("table01.csv", header=None)
years = df.iloc[0]
years = pd.to_numeric(years).astype(int).tolist()
RVA = df.iloc[1].astype(int).tolist()
plt.plot(years, RVA, marker="o")
plt.title("RVA vs Years")
plt.xlabel("Years")
plt.ylabel("RVA in Millions Dollars")
plt.show()

adf_test = adfuller(RVA)
print('ADF Val: %f' % adf_test[0])
print('p-value: %f' % adf_test[1])

# We may use this code to find the parameter of the ARIMA model if we have larger or more compelx datasets.
# However, for this dataset, we are going to use the parameter of ARIMA(trainDf, order=(1, 0, 0))
# because the dataset is too small. The plot_acf and plot_pacf functions could be used if we have larger datasets,
# then in that case the "ARIMA(p,d,q)" could be determined by "acf or pacf graph". In PACF graph, we can determine p in which
# the lag is significant; in ACF, we can determine q.
# plot_acf(RVA, lags=3)
# plot_pacf(RVA, lags=3)
# plt.show()

# the code for predicting only one year, which is 2023
# model = ARIMA(RVA, order=(1, 0, 0))
# model_fit = model.fit()
#
# pre = model_fit.get_forecast(steps=1)
# print(pre.predicted_mean)
# print(pre.conf_int())
# print(model_fit.summary())

pre = []
for i in range(9, len(RVA)):
    trainDf = RVA[:i]
    testT = int(years[i])
    actual = float(RVA[i])
    mFit = ARIMA(trainDf, order=(1, 0, 0)).fit()
    preRes = float(mFit.forecast(steps=1)[0])
    d = abs(preRes - actual)
    pre.append((testT, actual, preRes, d/actual))
    print(mFit.summary())

res = pd.DataFrame(pre, columns=["Year", "Actual", "Predicted-Val", "Relative Difference"])
print(res.to_csv())