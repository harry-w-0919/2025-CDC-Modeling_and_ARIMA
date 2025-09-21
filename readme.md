# Predicting the Future Value of Real Value Added (RVA) for Manufacturing Category
### Model: Autoregressive Integrated Moving Average (ARIMA)

### 1. Python Environment Requirement
Platform: macOS 26.0 (25A354), Apple Silicon
1) Conda Version: conda 24.11.3
2) pandas 2.2.2
3) matplotlib 3.8.4
4) statsmodels 0.14.2

### 2. Data We Used (CSV File)
| Year | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 |
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| RVA  | 24193 | 26789 | 27121 | 29264 | 32387 | 32514 | 33589 | 37565 | 35618 | 38519 | 38760 | 38773 |

### 3. Breaking Down our Code

1) Differencing in ARIMA <br>
    Because we do not want our model to simply assume that as time goes on, the RVA is going to absolutely increase, 
we need to apply differencing. This concern was concluded from the "Time versus RVA Graph", which shows a increasing tendency.
Differencing helps remove this trend in order to let the model to focus on short-term changes within our dataset rather than 
simply focusing on long-term trends. <br>
    We need to perform a *Two-tailed Hypothesis Test*, where <br>
    &nbsp; &nbsp; H<sub>0</sub>: The data may exhibit a long-term trend, <br>
    &nbsp; &nbsp; H<sub>1</sub>: The data does not include a long-term trend. <br>
    We do this by the following code in our python file: <br>
    ```python
    adf_test = adfuller(RVA)
    print('ADF Val: %f' % adf_test[0])
    print('p-value: %f' % adf_test[1])
    ```
    Then, we have the output of: <br>
    ```text
    ADF Val: -3.190440
    p-value: 0.020545
    ```
    Because p-value ≤ 0.05, we reject H<sub>0</sub> and conclude that there is no convincing statistical evidence that
    the data may exhibit a long-term trend. Therefore, differencing is not needed. <br>

2) Unpacking ARIMA Model Parameters <br>
    We may use this code to find the parameter of the ARIMA model if we have larger or more compelx datasets.
However, for this dataset, we use the parameter of ARIMA(trainDf, order=(1, 0, 0))
because the dataset is small and we do not want to overcomplicate the model. <br> The "plot_acf" and "plot_pacf" functions could be used if we have larger datasets,
then in that case the "ARIMA(p,d,q)" could be determined by "acf or pacf graph". In "PACF graph", we can determine "p" in which
the lag is significant; in "ACF graph", we can determine "q".
    ```python
    # plot_acf(RVA, lags=3)
    # plot_pacf(RVA, lags=3)
    # plt.show()
    ARIMA(trainDf, order=(1, 0, 0)).fit()
    ```
3) ARIMA Validation <br>
    Because this dataset is small, and because we want to make sure that the ARIMA works on annually data prediction, we
implement out-of-sample validation by reserve some of the real-world data and walk-forward validation. Overall, this looks like: <br>
    &nbsp; &nbsp; *a) use 2012-2020 data to predict RVA for 2021, then* <br>
    &nbsp; &nbsp; *b) use 2012-2021 data to predict RVA for 2022, then* <br>
    &nbsp; &nbsp; *c) use 2012-2022 data to predict RVA for 2023.* <br>
    We do have the real-world data for the year 2021, 2022, and 2023, so we can directly compare the data from the prediction with
our actual values.

### 4. ARIMA Results and Analysis
Specifically, when predicting the RVA of year 2023, we have the model test-statistics output, which can be use to determine the fit of the model. <br>

| Statistic              | Value | p-value | Interpretation                                                                  |
|-------------------------|-------|---------|---------------------------------------------------------------------------------|
| Jarque-Bera (JB)       | 0.49  | 0.78    | Residuals close to normal distribution                                          |
| Heteroskedasticity (H) | 1.84  | 0.57    | No significant heteroskedasticity found in residual, making the variance stable |


| Year | Actual  | Predicted-Val | Relative Difference |
|:----:|:-------:|:-------------:|:-------------------:|
| 2021 | 38519.0 | 35123.68      | 0.0881              |
| 2022 | 38760.0 | 37998.93      | 0.0196              |
| 2023 | 38773.0 | 38355.56      | 0.0108              |

**Note: "Relative Differences" are calculated by (abs(actual - predicted)/predicted).** <br> <br>
From the results above, we can see that the ARIMA model is able to predict relatively accurate results by calculating the relative differences
with the actual RVA values. Therefore, we can conclude that the ARIMA model can provide numerical results that will be useful for investing advisor
to offer reliable advice to investors. However, some really accurate predictions in our model may come from the limited dataset. <br> In the future, if we have
more reliable and detailed datasets that we can use, we could refer to the same process to create the ARIMA model and yield better results for audiences
who might need them. <br> Overall, our project showcases that the ARIMA model has potentials in predicting data with time related investing problems.
