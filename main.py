import numpy as np
import pandas
import pandas as pd
from pandas import read_csv, DataFrame, Series
from matplotlib import pyplot as plt
from numpy import polyfit
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit(X, y, degree=3):
    coef = polyfit(X, y, degree)
    trendpoly = np.poly1d(coef)
    return trendpoly(X)


def get_season(s, index, months=12, degree=3):
    X = [i % months for i in range(0, len(s))]
    seasonal = fit(X, s.values, degree)
    return Series(data=seasonal, index=index)


def get_trend(s, index, degree=3):
    X = list(range(len(s)))
    trend = fit(X, s.values, degree)
    return Series(data=trend, index=index)


def load_df(f):
    df = read_csv(
        './spark_wines/wine_Austral.dat',
        header=0,
        delimiter='\t',
        parse_dates=['date_'],
        index_col=['date_']
    )

    def inner():
        f(df)

    return inner


@load_df
def plot_graph(df: DataFrame):
    plt.plot(df.index.values, df["spark"])
    plt.show()


@load_df
def plot_extended(df: DataFrame):
    df["trend"] = get_trend(df["spark"], df.index)
    df["season"] = get_season(df["spark"], df.index)
    plt.plot(df.index.values, df["spark"])
    plt.plot(df.index.values, df["trend"], color="red")
    plt.plot(df.index.values, df["season"], color="green")
    plt.grid(axis='x')
    # plt.xticks(df.index.values, rotation=90)
    plt.show()


#
# def ARMA_prediction(plt, train, test):
#     y = train
#     ARMAmodel = SARIMAX(y, order=(1, 0, 1))
#     ARMAmodel = ARMAmodel.fit()
#     y_pred = ARMAmodel.get_forecast(len(test.index))
#     y_pred_df = y_pred.conf_int(alpha=0.05)
#     y_pred_df["Predictions"] = ARMAmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
#     y_pred_df.index = test.index
#     y_pred_out = y_pred_df["Predictions"]
#     plt.plot(y_pred_out, color='green', label='ARMA Predictions')
#
#
# def ARIMA_prediction(plt, train, test):
#     y = train
#     ARIMAmodel = ARIMA(y, order=(2, 3, 7))
#     ARIMAmodel = ARIMAmodel.fit()
#
#     y_pred = ARIMAmodel.get_forecast(len(test.index))
#     y_pred_df = y_pred.conf_int(alpha=0.05)
#     y_pred_df["Predictions"] = ARIMAmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
#     y_pred_df.index = test.index
#     y_pred_out = y_pred_df["Predictions"]
#     plt.plot(y_pred_out, color='Yellow', label='ARIMA Predictions')
#
# def AR_prediction(plt, train, test):
#     y = train
#     ARmodel = AutoReg(y, lags=1)
#     ARmodel = ARmodel.fit()
#
#     # y_pred = ARmodel.get_forecast(len(test.index))
#     # y_pred_df = y_pred.conf_int(alpha=0.05)
#     ARmodel["Predictions"] = ARmodel.predict(start=len(test.index), end=len(test.index))
#     ARmodel.index = test.index
#     y_pred_out = ARmodel["Predictions"]
#     plt.plot(y_pred_out, color='Yellow', label='ARIMA Predictions')
#     plt.legend()

def FA_predict(plt, train, test):
    forecaster = ForecasterAutoreg(
        regressor=RandomForestRegressor(random_state=123),
        lags=20
    )
    forecaster.fit(y=train)
    steps = 18
    predictions = forecaster.predict(steps=steps)

    df = pd.DataFrame(
        {
            "Train": test.values,
            "Predict": predictions.values,
        },
        index=test.index,
    )
    plt.plot(test.index, predictions, color='Green', label='Predictions')
    print(df)


@load_df
def plot_prediction(df: DataFrame):
    train = df[df.index < pandas.to_datetime("1993-01-01", format='%Y-%m-%d')]["spark"]
    test = df[df.index >= pandas.to_datetime("1993-01-01", format='%Y-%m-%d')]["spark"]
    plt.plot(train, color="black")
    plt.plot(test, color="red")
    plt.xticks(rotation=45)

    FA_predict(plt, train, test)

    plt.legend()
    plt.show()


def main():
    # plot_extended()
    plot_prediction()


if __name__ == '__main__':
    main()
