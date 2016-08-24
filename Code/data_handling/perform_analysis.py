import datetime
import pickle

import numpy as np
import pandas as pd

from data_handling import db_handling, sentiment, stock_quotes


def is_monday_to_thursday(dt):
    return dt.weekday() <= 4

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

def perform_daily_analysis(start_dt, end_dt, stock_symbol, tweets_collection, log_file_path = None, pickle_file_path = None):

    log_file = setup_logfile(log_file_path)

    from_dt_array, to_dt_array, sentiment_array, stock_yield_array = [], [], [], []

    from_dt = start_dt
    to_dt = start_dt + datetime.timedelta(days=1)

    while to_dt < end_dt:

        if is_monday_to_thursday(from_dt):
            try:
                #calculate stock yield and sentiment
                day_stock_yield = calc_stock_yield(from_dt, to_dt, stock_symbol)
                day_sentiment = calc_day_sentiment(from_dt, to_dt, stock_symbol, tweets_collection)

                #append current values to array
                from_dt_array.append(from_dt)
                to_dt_array.append(to_dt)
                stock_yield_array.append(day_stock_yield)
                sentiment_array.append(day_sentiment)

                log("Analysis Successful: " + str([from_dt, to_dt, day_stock_yield, day_sentiment]), log_file)
            except BaseException as e:
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
    correlation = np.corrcoef(df["sentiment"], df["yield"])[0, 1]

    if pickle_file_path is not None:
        pickle.dump(pd_dataframe, open(pickle_file_path, "wb"))

    return pd_dataframe, correlation