class Tweets_Statistic:

	def __init__(self, tweets=[]):
		self.number_of_tweets = 0
		self.mean_follower_count = 0
		self.mean_number_of_stock_symbols = 0
		self.mean_number_of_urls = 0
		self.symbol_tally = {}


	def update_symbol_tally(self, tweet):
		for s in tweet.symbols:
			if s in self.symbol_tally:  # stock symbol already exists in tally
				self.symbol_tally[s] += 1
			else:  # add stock symbol to tally
				self.symbol_tally[s] = 1

	def add_tweet_to_statistic(self, tweet):
		n_old = self.number_of_tweets
		self.number_of_tweets += 1
		n_new = self.number_of_tweets

		# update basic statistics
		self.mean_follower_count = ((self.mean_follower_count * n_old) + tweet.follower_count) / float(n_new)
		self.mean_number_of_stock_symbols = ((self.mean_number_of_stock_symbols * n_old) + tweet.number_of_symbols()) / float(n_new)
		self.mean_number_of_urls = ((self.mean_number_of_urls * n_old) + tweet.number_of_urls()) / float(n_new)

		#update symbol tally
		self.update_symbol_tally(tweet)

	def __str__(self):
		s = "Number of Tweets considered: " + str(self.number_of_tweets) + "\n"
		s += "Mean Number Of Followers: " + str(self.mean_follower_count) + "\n"
		s += "Mean Number of Stock Symbols: " + str(self.mean_number_of_stock_symbols) + "\n"
		s += "Mean Number of URLs: " + str(self.mean_number_of_urls) + "\n"
		s += "Symbol Tally: " + str(self.symbol_tally)
		return s