from pymongo import MongoClient
import numpy as np
import time
import datetime
import os
import bson
from tweet import Tweet
from parser import Parser
from sentiment import sentiment
from tweets_statistic import Tweets_Statistic
import pickle
from pprint import pprint


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

def convert_db_timestamps_to_int(collection, as_bulk = False):
    cursor = collection.find()
    count = cursor.count()
    n = 0
    if not as_bulk:
        for tweet in cursor:
            if n % 1000 == 0:
                print(n)
            id = tweet["id"]
            timestamp = int(tweet["timestamp_ms"])
            collection.update_one({"id": id}, {'$set': {'timestamp_ms': timestamp}})
            n += 1
    else:
        if n % 1000 == 0:
            bulk = collection.initialize_ordered_bulk_op()
        id = tweet["id"]
        timestamp = int(tweet["timestamp_ms"])
        bulk.find({"id": id}).update({'$set': {'timestamp_ms': timestamp}})
        if n == (count-1) or (n % 10 == 9):
            print("Execute Bulk Operation")
            result = bulk.execute()
            pprint(result)
            print("----> N = " + str(n))
        n += 1

def tweet_query(collection, start_date, end_date, symbol):
    return collection.find({"timestamp_ms": {'$gte': start_date, '$lt': end_date}})

if __name__ == '__main__':
    # connect to db_collection
    db = connect_twitter_db()
    tweets_db_collection = get_tweets_collection(db)
    prices_collection = get_prices_collection(db)

    convert_db_timestamps_to_int(tweets_db_collection)

    '''
    t1 = tweets_db_collection.find()[1]
    t2 = tweets_db_collection.find()[20000]

    tweet1 = Tweet(t1)
    tweet2 = Tweet(t2)
    print(tweet1)

    stat = Tweets_Statistic()
    stat.add_tweet_to_statistic(tweet1)
    stat.add_tweet_to_statistic(tweet2)
    print(stat)
    '''