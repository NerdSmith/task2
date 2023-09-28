import numpy as np
from pandas import read_csv, DataFrame, Series
from matplotlib import pyplot as plt
from numpy import polyfit


def fit(X, y, degree=3):
    coef = polyfit(X, y, degree)
    trendpoly = np.poly1d(coef)
    return trendpoly(X)


def get_season(s, index,  months=12, degree=3):
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
    plt.show()

def main():
    plot_extended()


if __name__ == '__main__':
    main()
