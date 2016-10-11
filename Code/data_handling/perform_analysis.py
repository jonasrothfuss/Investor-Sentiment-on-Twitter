import datetime
import pickle
import scipy
import pytz
from collections import deque
import traceback
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import pandas as pd

from data_handling import db_handling, sentiment, stock_quotes
import bag_of_words_model

def is_monday_to_thursday(dt):
    return dt.weekday() < 4

def is_monday_to_friday(dt):
    return dt.weekday() <= 5

def is_monday(dt):
    return dt.weekday() == 1


def calc_stock_yield(from_dt, to_dt, stock_symbol):
    return stock_quotes.open_to_open_yield(stock_symbol, from_dt, to_dt)

def calc_day_sentiment(from_dt, to_dt, stock_symbol, tweets_collection):
    cursor = db_handling.tweet_query(tweets_collection, from_dt, to_dt)
    tweet_array = db_handling.tweets_as_object_array(cursor, stock_symbol)
    return sentiment.bulk_sentiment(tweet_array)

def log(log_message, log_file):
    if log_file is None:
        print(log_message)
    else:
        log_file.write(log_message + '\n')

def setup_logfile(log_file_path):
    if log_file_path is not None:
        return open(log_file_path, "w+")
    else:
        return None

def perform_daily_analysis(start_dt, end_dt, stock_symbol, tweets_df, log_file_path = None, pickle_file_path = None):

    log_file = setup_logfile(log_file_path)

    from_dt_array, to_dt_array, sentiment_array, stock_yield_array = [], [], [], []

    from_dt = start_dt
    to_dt = start_dt + datetime.timedelta(days=1)

    while to_dt < end_dt:

        if is_monday_to_thursday(from_dt):
            try:
                #calculate stock yield and sentiment
                day_stock_yield = calc_stock_yield(from_dt, to_dt, stock_symbol)
                day_sentiment = bag_of_words_model.bulk_sentiment_twitter(from_dt, to_dt, stock_symbol, tweets_df)

                #append current values to array
                from_dt_array.append(from_dt)
                to_dt_array.append(to_dt)
                stock_yield_array.append(day_stock_yield)
                sentiment_array.append(day_sentiment)

                log("Analysis Successful: " + str([from_dt, to_dt, day_stock_yield, day_sentiment]), log_file)
            except BaseException as e:
                traceback.print_exc(e)
                log("--- Could not finish analysis in time period " + str(from_dt) + " - " + str(to_dt) + ": " + str(e), log_file)

        #go to next day
        from_dt = from_dt + datetime.timedelta(days=1)
        to_dt = to_dt + datetime.timedelta(days=1)

    if log_file_path is not None:
        log_file.close()


    pd_dataframe = pd.DataFrame({'from_dt': from_dt_array,
                                 'to_dt': to_dt_array,
                                 'stock_yield': stock_yield_array,
                                 'sentiment': sentiment_array
                                 })

    df = pd_dataframe[pd_dataframe["stock_yield"] != 0]
    correlation, p_val = scipy.stats.pearsonr(df["sentiment"], df["stock_yield"])


    if pickle_file_path is not None:
        pickle.dump(pd_dataframe, open(pickle_file_path, "wb"))

    return pd_dataframe, correlation, p_val


def df_for_granger(tweets_df, ticker):
    start_dt = datetime.datetime(2015, 10, 23, 21, 0, 0, 0, pytz.UTC)
    end_dt = datetime.datetime(2016, 6, 24, 21, 0, 0, 0, pytz.UTC)

    fin_data_dict = pickle.load(open('../Data/Finance_Quotes/fin_data_dict.pickle', 'rb'))[ticker]
    fin_data_dict.index = fin_data_dict['Date']
    print(fin_data_dict)

    date_array, sentiment_array, stock_close_array = [], [], []

    from_dt = start_dt
    to_dt = start_dt + datetime.timedelta(days=1)

    while to_dt < end_dt:

        if is_monday_to_friday(to_dt):
            try:
                # calculate stock yield and sentiment
                if is_monday(to_dt): #calculate sentiment score over entire weekend
                    day_sentiment = bag_of_words_model.bulk_sentiment_twitter(to_dt - datetime.timedelta(days=3), to_dt, tweets_df, ticker)
                    day_stock_close = fin_data_dict['Close'][stock_quotes.datetime_to_str(to_dt)] - fin_data_dict['Close'][stock_quotes.datetime_to_str(to_dt - datetime.timedelta(days=3))]
                else:
                    day_sentiment = bag_of_words_model.bulk_sentiment_twitter(from_dt, to_dt, tweets_df, ticker)
                    day_stock_close = fin_data_dict['Close'][stock_quotes.datetime_to_str(to_dt)] - fin_data_dict['Close'][stock_quotes.datetime_to_str(from_dt)]

                # append current values to array
                date_array.append(to_dt.date())
                stock_close_array.append(day_stock_close)
                sentiment_array.append(day_sentiment)

                print("Analysis Successful: " + str([from_dt, to_dt, day_stock_close, day_sentiment]))
            except BaseException as e:
                print("--- Could not finish analysis in time period " + str(from_dt) + " - " + str(to_dt) + ": " + str(e))

        # go to next day
        from_dt = from_dt + datetime.timedelta(days=1)
        to_dt = to_dt + datetime.timedelta(days=1)

    close_lag_1 = deque(stock_close_array)
    close_lag_1.pop()
    close_lag_1.appendleft(np.nan)

    pd_dataframe = pd.DataFrame({'date': date_array,
                                 'close': stock_close_array,
                                 'sentiment': sentiment_array
                                 })
    #add lags
    close_lag = deque(stock_close_array)
    for i in range(1, 7):
        close_lag.pop()
        close_lag.appendleft(np.nan)
        pd_dataframe['close_lag_' + str(i)] = list(close_lag)

    sentiment_lag = deque(sentiment_array)
    for i in range(1, 7):
        sentiment_lag.pop()
        sentiment_lag.appendleft(np.nan)
        pd_dataframe['sentiment_lag_' + str(i)] = list(sentiment_lag)

    return pd_dataframe

def granger_analysis(granger_df):
    sm.tsa.stattools.grangercausalitytests(granger_df.as_matrix(['sentiment','close']), 6)
    #result = sm.ols(formula="close ~ close_lag_1 + close_lag_2 + close_lag_3 + close_lag_4 + close_lag_5 + close_lag_6 " +
    #                        "+ sentiment_lag_1 + sentiment_lag_2 + sentiment_lag_3 + sentiment_lag_4 + sentiment_lag_5 + sentiment_lag_6", data=granger_df).fit()
    #print(result.summary())

def granger_plot(granger_df):
    plt.style.use('ggplot')
    plt.plot(granger_df['date'], scipy.stats.mstats.zscore(granger_df['sentiment']), linewidth=2.0)
    plt.plot(granger_df['date'], scipy.stats.mstats.zscore(granger_df['close']), linewidth=2.0)
    plt.legend(['Sentiment Score', 'Movement DJIA'], loc='upper right', fontsize=12)
    plt.ylabel('z-score', fontsize=14)
    plt.show()
