import calendar
import datetime
import os
import pickle
import time
from pprint import pprint

import numpy as np
import pandas as pd
import theano
from dateutil import parser
from pymongo import MongoClient
from theano import tensor as T

from RNTN.rntn import RNTN
from data_handling.tweet import Tweet

Dow_Jones_Tickers = {'MMM': '3M', 'AXP': 'American Express', 'AAPL': 'Apple', 'BA': 'Boeing', 'CAT': 'Caterpillar',
                     'CVX': 'Chevron', 'CSCO': 'Cisco', 'KO': 'Coca Cola', 'DIS': 'Disney', 'DD': 'Du pont de Nemours',
                     'XOM': 'Exxon Mobil', 'GE': 'General Electrics', 'GS': 'Goldman Sachs', 'HD': 'Home Depot',
                     'IBM': 'IBM', 'INTC': 'Intel', 'JNJ': 'Johnson & Johnson', 'JPM': 'JPMorgan Chase', 'MCD': 'McDonald\'s',
                     'MRK': 'Merck', 'MSFT': ' Microsoft', 'NKE': 'Nike', 'PFE': 'Pfizer', 'PG': 'Proctor & Gamble',
                     'TRV': 'Travlers Companies Inc', 'UTX': 'United Technologies', 'UNH': 'UnitedHealth', 'VZ': 'Verizon',
                     'V': 'Visa', 'WMT': 'Wal-Mart'}

def print_dict(dict):
    for key in dict.keys():
        print(key + ": " + str(dict[key]))

# returns collection object from MongoDB
def connect_twitter_db():
    client = MongoClient()
    db = client.twitter_db
    return db

def get_tweets_collection(db):
    print(type(db))
    print("Tweets Collection Size: " + str(db.command("collstats", "tweets")["storageSize"] / 1000000.0) + " MB")
    return db.tweets

def get_prices_collection(db):
    print("Tweets Collection Size: " + str(db.command("collstats", "prices")["storageSize"] / 1000000.0) + " MB")
    return db.prices

def random_tweets_sample(collection, sample_size):
    # generate random sample of indexes
    collection_size = collection.find().count()
    random_indexes = np.random.randint(0, collection_size, sample_size)

    # retrieve documents corresponding to the indexes
    tweets_sample = []
    cursor = collection.find()
    for i in random_indexes:
        tweets_sample.append(cursor[int(i)])
    return tweets_sample

def db_collection_as_array(db_collection):
    tweets_array = []
    cursor = db_collection.find(no_cursor_timeout=True).batch_size(1000)
    collection_size = cursor.count()
    count = 1
    timestamp = time.clock()
    for t in cursor:
        tweets_array.append(t)
        count += 1
        if (count % 50000) == 0:
            print('' + str(count / float(collection_size) * 100.0) + " %    " + "Duration: " + str(
                time.clock() - timestamp))
            timestamp = time.clock()
    cursor.close()
    return tweets_array

#provides the respective stockprice at the next full minute
def get_stock_price(collection, symbol, dt):
    datetime_str = format_datetime_for_query(dt)
    cursor = collection.find({"ticker": symbol, "datetime_utc": datetime_str})
    return cursor[0]['price']

def format_datetime_for_query(dt):
    return str(round_to_next_minute(dt).strftime("%Y-%m-%d %H:%M:%S")) + " UTC"

def round_to_next_minute(dt):
    if dt.second == 0:
        return dt
    else:
        return dt + datetime.timedelta(seconds=60-dt.second)

def percentage_tweets_with_in_time_stock_price(tweets_collection,prices_collection, number_of_tweets):
    s = tweets_collection.find()
    s.batch_size(2000)
    count = 0
    success_count = 0

    for i in s:
        if count % 4000 == 0:
            print("----------- Count: " + str(count) + "  success_count: " + str(success_count))
        if count > number_of_tweets:
            break
        try:
            t = Tweet(i)
            print(
                str(get_stock_price(prices_collection, t.symbols[0], t.created_at)) + "  " + str(t.created_at) + "  " +
                t.symbols[0])
            success_count += 1
        except Exception:
            print("ERROR: " + str(t.created_at) + "  " + str(t.symbols))
        count += 1

    return success_count/float(count)

def write_tweets_to_file(tweets_collection, filepath, force=False, limit = 50000000):
    cursor = tweets_collection.find()
    cursor.batch_size(1000)
    count = 0
    if os.path.exists(filepath) and not force:
        # override by setting force=True.
        print('File already present - Skipping pickling.')
    else:
        try:
            with open(filepath, 'wb') as f:
                for t in cursor:
                    if count > limit:
                        break
                    count += 1
                    pickle.dump(Tweet(t), f, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data', ':', e)

def load_tweets_from_file(file_path):
    tweet_array = []
    with open(file_path, 'rb') as f:
        for _ in range(pickle.load(f)):
            tweet_array.append(pickle.load(f))
    return tweet_array

def convert_db_timestamps_to_int(collection, as_bulk = False, bulk_size = 1000):
    cursor = collection.find().skip(872453)
    count = cursor.count()
    n = 872453

    if  as_bulk:
        number_updates_pending = 0
        bulk = collection.initialize_unordered_bulk_op()
        for tweet in cursor:
            if n % 1000 == 0:
                print("Tweets processed: " + str(n))
            try:
                id = tweet["id"]
                ts = tweet["timestamp_ms"]
                if (type(ts) is str):
                    ts_int = int(ts)
                    bulk.find({"id": id}).update({'$set': {'timestamp_ms': ts_int}})
                    number_updates_pending += 1
                    print("Items in bulk: " + str(number_updates_pending))
            except Exception as e:
                print(e)
                collection.delete_one({"_id": tweet['_id']})
                print("Object deleted")
            if number_updates_pending >= bulk_size or n == count - 1:
                pprint(bulk.execute())
                bulk = collection.initialize_unordered_bulk_op()
                number_updates_pending = 0
            n += 1
    else:
        t = time.clock()
        for tweet in cursor:
            if n % 1000 == 0:
                print(str(n) + "   Processing Time: " + str(time.clock()-t) + " sec")
                t = time.clock()
            try:
                id = tweet["id"]
                ts = tweet["timestamp_ms"]
                if (type(ts) is str):
                    #print(n)
                    timestamp = int(ts)
                    collection.update_one({"id": id}, {'$set': {'timestamp_ms': timestamp}})
                n += 1
            except Exception as e:
                print(e)
                collection.delete_one({"_id": tweet['_id']})
                print("Object deleted")

def tweet_query(collection, start_datetime, end_datetime):
    start_ts = datetime_to_ms_utc_timestamp(start_datetime)
    end_ts = datetime_to_ms_utc_timestamp(end_datetime)
    print(start_ts)
    print(end_ts)
    return collection.find({"timestamp_ms": {'$gte': start_ts, '$lt': end_ts}})

def datetime_to_ms_utc_timestamp(dt):
    return calendar.timegm(dt.utctimetuple()) * 1000

def tweets_as_object_array(cursor, symbol = None):
    tweet_array = []
    for t in cursor:
        tweet = Tweet(t)
        if (symbol is None) or tweet.has_symbol(symbol):
            tweet_array.append(tweet)
    return tweet_array

def datetime_from_str(dt_str):
    return parser.parse(dt_str)

def stock_prices_as_panda_df(collection):
    stock_symbols = Dow_Jones_Tickers.keys()
    first_symbol = True
    for symbol in stock_symbols:
        cursor = collection.find({"ticker": symbol})
        prices = []
        dts = []
        for p in cursor:
            prices.append(float(p['price']))
            dts.append(datetime_from_str(p['datetime_utc']))
        current_panda_frame = pd.DataFrame(np.transpose(np.array([dts, prices])), columns=['time', symbol])
        if not first_symbol:
            panda_frame = pd.merge(panda_frame, current_panda_frame, how='outer', on=['time', 'time'])
        else:
            panda_frame = current_panda_frame
            first_symbol = False
    return panda_frame.sort_values(by = 'time')

def load_stockprices_as_panda_df(pickle_file_path = '/home/jonasrothfuss/Dropbox/Eigene Dateien/Uni/Bachelorarbeit/DumpData/prices_pickle'):
    return pickle.load(open(pickle_file_path, 'rb'))

if __name__ == '__main__':
    # connect to db_collection
    db = connect_twitter_db()
    tweets_db_collection = get_tweets_collection(db)
    prices_collection = get_prices_collection(db)

    theano.config.floatX = 'float32'

    r = RNTN(20, 3)
    b = r.init_vector([2*r.emb_dim, 1])

    e1 = T.fcol('e1')
    e2 = T.fcol('e1')

    emb1 = np.random.normal(size=[20, 1]).astype(theano.config.floatX)
    emb2 = np.random.normal(size=[20, 1]).astype(theano.config.floatX)

    ru = r.create_recursive_unit()
    xpr = ru(e1, e2)
    print(theano.pp(xpr))

    f = theano.function([e1,e2], xpr)

    y = f(emb1, emb2)

    print(y[0].shape)
    print(y)

    #start_dt = datetime.datetime(2015, 10, 22, 13, 30, 0)
    #end_dt = datetime.datetime(2016, 10, 23, 13, 30, 0)
    #perform_analysis.perform_daily_analysis(start_dt, end_dt, 'IBM', tweets_db_collection)


    #(result)
    '''
    cursor = tweets_db_collection.find()
    print(cursor[10000]["timestamp_ms"])
    t1 = Tweet(cursor[10000])


    q1 = tweet_query(tweets_db_collection, start_dt, end_dt)
    print(q1.count())
    '''
