from dateutil.parser import parse

def stock_symbols(tweet_dict):
	symbols = []
	for s in tweet_dict['entities']['symbols']:
		symbols.append(s['text'])
	return symbols

class Tweet:

	def __init__(self, tweet_dict):
		self.text = tweet_dict["text"]
		self.created_at = parse(tweet_dict['created_at'])
		self.language = tweet_dict['lang']
		self.retweet_count = tweet_dict["retweet_count"]
		self.urls = tweet_dict['entities']['urls']
		self.hashtags = tweet_dict['entities']['hashtags']
		self.symbols = stock_symbols(tweet_dict)
		self.follower_count = tweet_dict['user']['followers_count']
		self.user = tweet_dict['user']

	def number_of_urls(self):
		return len(self.urls)

	def number_of_symbols(self):
		return len(self.symbols)

	def __str__(self):
		s = "Date: " + str(self.created_at) + "\n"
		s += "Text: " + str(self.text) + "\n"
		s += "Symbols: " + str(self.symbols) + "\n"
		return s

