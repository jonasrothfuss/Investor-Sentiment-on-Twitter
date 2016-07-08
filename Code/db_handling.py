from pymongo import MongoClient

#connect to MongoDB
client = MongoClient()
db = client.twitter_db
tweets = db.finance_collection

t1 = tweets.find_one()
print("Hello World")
print(t1['text'])
