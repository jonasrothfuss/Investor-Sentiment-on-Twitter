from pymongo import MongoClient
import datetime
from dateutil.parser import parse

#returns collection object from MongoDB
def connect_twitter_db():
	client = MongoClient()
	db = client.twitter_db
	return db.finance_collection

def print_dict(dict):
	for key in dict.keys():
		print(key + ": " + str(dict[key]))

def get_tweet_datetime(tweet):
	return parse(tweet['created_at'])

def get_follower_count(tweet):
	return tweet['user']['followers_count']

def get_stock_symbols(tweet):
	symbols = []
	for s in tweet['entities']['symbols']:
		symbols.append(s['text'])
	return symbols

#connect to MongoDB
tweets = connect_twitter_db()

#get first tweet
t1 = tweets.find_one()

print(get_tweet_datetime(t1))

print(get_stock_symbols(t1))



'''
for post in tweets.find({"date": {"$lt": d}}).sort("author"):
...   print post

'''
