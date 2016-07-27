from pymongo import MongoClient
import datetime
import numpy as np
import time
from dateutil.parser import parse

def print_dict(dict):
	for key in dict.keys():
		print(key + ": " + str(dict[key]))

#returns collection object from MongoDB
def connect_twitter_db():
	client = MongoClient()
	db = client.twitter_db
	print("Collection Size: " + str(db.command("collstats", "finance_collection")["storageSize"]/1000000.0) + " MB")
	return db.finance_collection

def random_tweets_sample(collection, sample_size):
	#generate random sample of indexes
	collection_size = collection.find().count()
	random_indexes = np.random.randint(0,collection_size,sample_size)
	print(random_indexes)
	print(type(random_indexes[1]))

	#retrieve documents corresponding to the indexes
	tweets_sample = []
	for i in random_indexes:
		tweets_sample.append(collection.find()[int(i)])
	return tweets_sample

def db_collection_as_array(db_collection):
    tweets_array = []
    cursor = db_collection.find(no_cursor_timeout = True).batch_size(1000)
    collection_size = cursor.count()
    count = 1
    timestamp = time.clock()
    for t in cursor:
        tweets_array.append(t)
        count += 1
        if (count % 50000) == 0:
            print('' + str(count/float(collection_size)*100.0) + " %    " + "Duration: " + str(time.clock()-timestamp))
            timestamp = time.clock()
    cursor.close()
    return tweets_array

def update_stock_symbol_statistic(stock_symbol_tally, tweet):
	for s in stock_symbols(tweet):
		if s in stock_symbol_tally: #stock symbol already exists in tally
			stock_symbol_tally[s] += 1
		else: #add stock symbol to tally
			stock_symbols[s] = 1

def generate_tweet_statistic(tweets):
	number_of_tweets = len(tweets)
	mean_follower_count, mean_number_of_stock_symbols, mean_number_of_urls = 0
	stock_symbol_tally = {}
	for t in tweets:
		mean_follower_count += follower_count(t)
		mean_number_of_urls += number_of_urls(t)
		stock_symbol_tally = update_stock_symbol_statistic(stock_symbol_tally, t)









if __name__ == '__main__':
	#connect to db_collection
	tweets_db_collection = connect_twitter_db()

	t1 = tweets_db_collection.find()[1]
	t2 = tweets_db_collection.find()[20000]

	print(number_of_urls(t1))
	print(number_of_urls(t2))

	print_dict(t2)
