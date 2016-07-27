from pymongo import MongoClient
import numpy as np
import time
from tweet import Tweet
from tweets_statistic import Tweets_Statistic


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

if __name__ == '__main__':
	#connect to db_collection
	tweets_db_collection = connect_twitter_db()

	t1 = tweets_db_collection.find()[1]
	t2 = tweets_db_collection.find()[20000]

	tweet1 = Tweet(t1)
	tweet2 = Tweet(t2)
	print(tweet1)


	stat = Tweets_Statistic()
	stat.add_tweet_to_statistic(tweet1)
	stat.add_tweet_to_statistic(tweet2)
	print(stat)