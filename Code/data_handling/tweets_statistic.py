import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from data_handling import db_handling
import pandas as pd
import numpy as np
import operator


def ticker_histogram(tweets_df, number_of_tickers = 30):
    ticker_tally = {}
    number_of_tweets = len(tweets_df.index)
    for ticker in db_handling.Dow_Jones_Tickers.keys():
        ticker_tally[ticker] = len(tweets_df[tweets_df[ticker]==True][ticker])/float(number_of_tweets)

    tickers, values = dict_as_lists(ticker_tally, max_number=number_of_tickers)
    plt.style.use('ggplot')
    plt.bar(range(len(values)), values, align='center', color ='grey')
    plt.xticks(range(len(tickers)), tickers)
    plt.xlim((-1, number_of_tickers +2))
    plt.ylabel('percentage of tweets with respective ticker' + '\n')
    #plot.yaxis.set_major_formatter(FuncFormatter('%.0f%%'))
    plt.show()

def colsum_ticker_occurance_count(tweets_df):
    tweet_occurance_array = tweets_df.as_matrix(columns=db_handling.Dow_Jones_Tickers.keys())
    return np.sum(tweet_occurance_array, axis=1)

def simultaneous_ticker_occurance_diagram(tweets_df):
    colsum = colsum_ticker_occurance_count(tweets_df)
    plt.hist(colsum, 25, normed=1)
    plt.show()
    #print(set(list(np.sum(tweet_occurance_array, axis = 1)))

def histo_normdist_plot(data_array, xrange, xlabel = None, ylabel = None):
    plt.style.use('ggplot')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    _, bins, patches = ax.hist(data_array, 50, range=xrange, normed=True, facecolor='grey')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    #normal dist
    sd = np.std(data_array)
    mu = np.mean(data_array)
    print('sd: ', sd, 'mu ', mu)
    bincenters = 0.5 * (bins[1:] + bins[:-1])
    y_norm = mlab.normpdf(bincenters, mu, sd)
    ax.plot(bincenters, y_norm, 'r--', linewidth=1, color = 'black')
    if xlabel:
        plt.xlabel(xlabel, fontsize=18)
    if ylabel:
        plt.ylabel(ylabel, fontsize=18)

    plt.show()

def tweets_with_one_ticker(tweets_df):
    colsum = colsum_ticker_occurance_count(tweets_df)
    return tweets_df[colsum == 1]

def dict_as_lists(dict, descending = True, max_number = None):
    ordered_tuples = sorted(dict.items(), reverse=descending, key=operator.itemgetter(1))
    key_array = []
    value_array =[]
    for n, tuple in enumerate(ordered_tuples):
        key_array.append(tuple[0])
        value_array.append(tuple[1])
        if max_number and n-1 >= max_number:
            break
    return key_array, value_array

def momentum_adadelta_performance_comp_plot():
    dev_loss_0 = [0.722225, 0.721471, 0.718549, 0.715227, 0.711542, 0.707376, 0.702472, 0.696637, 0.689834, 0.682137]
    dev_loss_1 = [0.716518, 0.701954, 0.690057, 0.66859, 0.636063, 0.590728, 0.53733, 0.469327, 0.404491, 0.338704]

    plt.style.use('ggplot')
    plt.plot(np.arange(1, 11), dev_loss_0, linewidth=2.0)
    plt.plot(np.arange(1, 11), dev_loss_1, linewidth=2.0)
    plt.ylim([0.0, 1.0])
    plt.xlabel('epoch')
    plt.ylabel('avg training loss')
    plt.legend(['Momentum SGD', 'AdaDelta'], loc='upper right')
    plt.show()

def vader_1h_lag_scatterplot(tweets_df):
    profit_array = []
    trade_strategy_array = []
    for score, lag in zip(tweets_df['vader_compound'], tweets_df['lag']):
        if score > 0.4:
            trade_strategy = 1.0
        elif score < -0.4:
            trade_strategy = -1.0
        else:
            trade_strategy = 0.0
        profit_array.append(trade_strategy * lag)
        trade_strategy_array.append(trade_strategy)
    print(np.mean(profit_array))
    plt.style.use('ggplot')
    plt.scatter(tweets_df['vader_compound'], tweets_df['lag'])
    plt.xlim([-1.0, 1.0])
    plt.ylim([-0.06, 0.06])
    plt.xlabel('VADER sentiment score', fontsize=18)
    plt.ylabel('1h stock price reaction', fontsize=18)
    plt.show()

def dev_accuaracy_plot():
    metrics_30min = pd.read_csv('../Data_Clean/result_dump/30min_lag/metrics.csv')
    metrics_1h = pd.read_csv('../Data_Clean/result_dump/1h_lag/metrics.csv')
    plt.style.use('ggplot')
    metrics_30min['dev accuracy'].plot(linewidth=2.0)
    metrics_1h['dev accuracy'].plot(linewidth=2.0)
    plt.legend(['30 min lag', '1 hour lag'], loc='upper right')
    plt.ylim([0.3, 0.45])
    plt.xlabel('', fontsize=16)
    plt.ylabel('accuracy', fontsize=16)
    plt.xticks(list(metrics_30min.index), ['', '', '', 'epoch 1', '', '', '', 'epoch 2', '', '', '', 'epoch 3', '', ''])
    plt.show()


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