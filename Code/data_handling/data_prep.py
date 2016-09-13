import nltk
from nltk.tree import Tree
from TreeLSTM import tree_rnn
from data_handling import parser
from data_handling import tweet
from data_handling import db_handling
from data_handling import tweets_statistic
from data_handling import stock_quotes
from data_handling import data_prep
import numpy as np
import pickle
import time
import datetime

def clean_tweets_in_df(s104dataframe):
    for i in range(len(s104dataframe["tweet"])):
        if i % 20000 == 0:
            print(i)
        cleaned_tweet = tweet.clean_tweet(s104dataframe.ix[i, 'tweet'])
        s104dataframe.set_value(i, 'tweet', cleaned_tweet)
    pickle.dump(s104dataframe, open(db_handling.sentiment104_PATH + 'train_cleaned.pickle', 'wb'))
    return s104dataframe

def normalize_sent_label(s104dataframe):
    label_translation_dict = {0: -1, 2: 0, 4: 1}
    for i in range(len(s104dataframe["label"])):
        if i % 20000 == 0:
            print(i)
        new_label = s104dataframe.ix[i, 'label'] + 1 #label_translation_dict[s104dataframe.ix[i, 'label']]
        s104dataframe.set_value(i, 'label', new_label)
    pickle.dump(s104dataframe, open(db_handling.sentiment104_PATH + 'train_cleaned.pickle', 'wb'))
    return s104dataframe

def build_binary_rnn_tree(tree, vocab, sent_label = None, subtree_root = None):
    if subtree_root:
        root = subtree_root
    else:
        root = tree_rnn.BinaryNode()
        root.label = int(sent_label)
    for t_node in tree:
        node = tree_rnn.BinaryNode()
        root.add_child(node)
        if isinstance(t_node, Tree) and t_node.height() >= 2:
            build_binary_rnn_tree(t_node, vocab, subtree_root = node)
        else: #leave node
            node.val = vocab.index(t_node.lower())
    return root

def build_rnn_trees(tweets, labels, vocab):
    tweets = list(tweets)
    labels = list(labels)
    assert len(tweets) == len(labels)

    data = []
    ts = time.clock()
    p = parser.Parser()
    for i in range(len(tweets)):
        t = tweet.clean_tweet(tweets[i])
        parse_tree = p.parse_tree(t, binary=True, preprocessed=True)

        tree = build_binary_rnn_tree(parse_tree, vocab, labels[i])
        data.append((tree, labels[i]))

        if i % 500 == 0:
            print(str(i) + "   Processing Time: " + str(time.clock()-ts) + ' sec')
            ts = time.clock()

        if i == 2000:
            break

    return data

def create_vocab(text_array, vocab_file_path = None):
    # extracts all words in an text_array and generates a vocabulary set
    # vocabulary set may be stored in a file

    p = parser.Parser()
    word_buffer = []

    n = 0
    ts = time.clock()
    for text_element in text_array:
        text = tweet.clean_tweet(text_element) #replace entities ushc as url in text
        word_buffer.extend(p.word_list(text))
        if n % 1000 == 0:
            print(str(n) + "   Processing Time: " + str(time.clock()-ts) + ' sec')
            ts = time.clock()
        n += 1

    vocab = set(word_buffer)

    if vocab_file_path:
        with open(vocab_file_path, 'w') as f:
            f.writelines(map(lambda x: x + "\n", vocab))

    return vocab

def filter_retweets(tweets_df):
    rt_bool_array = []
    for i in tweets_df.index:
        if 'rt' in tweets_df['text'][i]:
            rt_bool_array.append(False)
        else:
            rt_bool_array.append(True)
    return tweets_df[rt_bool_array]

def identify_ticker(tweets_df):
    ticker_array = []
    tickers = list(tweets_df.loc[:, db_handling.Dow_Jones_Tickers.keys()].columns.values)
    tweet_occurance_matrix = tweets_df.as_matrix(columns=db_handling.Dow_Jones_Tickers.keys())
    print(tweet_occurance_matrix.shape)
    assert len(tickers) == tweet_occurance_matrix.shape[1]
    for row in np.arange(tweet_occurance_matrix.shape[0]):
        for col in np.arange(tweet_occurance_matrix.shape[1]):
            if tweet_occurance_matrix[row, col]:
                ticker_array.append(tickers[col])
    assert len(ticker_array) == tweet_occurance_matrix.shape[0]
    return ticker_array

def round_to_minute(dt, floor = True):
    if dt.second == 0:
        return dt
    elif floor: #previous minute
        return dt - datetime.timedelta(seconds=dt.second)
    else: # ceil -> next minute
        return dt + datetime.timedelta(seconds=60 - dt.second)

def prepare_tweet_data(tweets_df):
    #only tweets that only have one ticker
    tweets_df = tweets_statistic.tweets_with_one_ticker(tweets_df)

    #filter retweets
    tweets_df = filter_retweets(tweets_df)

    #restructure df
    tweets_df['ticker'] = identify_ticker(tweets_df)
    tweets_df = tweets_df.drop(db_handling.Dow_Jones_Tickers, axis = 1)
    return tweets_df

def calulate_lag(dt, ticker, price_df, days = 0, hours = 0, minutes = 0):
    dt_0 = round_to_minute(dt, floor=True)
    assert dt_0.second == 0
    dt_1 = dt_0 + datetime.timedelta(days=days, hours=hours, minutes=minutes)
    p_0 = stock_quotes.get_price(ticker, dt_0, price_df)
    p_1 = stock_quotes.get_price(ticker, dt_1, price_df)
    if np.isnan(p_0) or np.isnan(p_0):
        lag = float('nan')
    else:
        lag = float(p_1)/float(p_0) - 1.0
    print(dt_0, lag)
    return lag

def add_lag_col_to_df(tweets_df, prices_df, days = 0, hours = 0, minutes = 0, delete_rows_with_nan_lag = False, dump_file_path = None):
    assert 'created_at' in tweets_df.columns and 'ticker' in tweets_df.columns
    lag_array = []

    for time_ticker_tuple in zip(tweets_df['created_at'], tweets_df['ticker']):
        dt = time_ticker_tuple[0]
        ticker = time_ticker_tuple[1]
        if dt.hour >= 13 and dt.hour <= 19:
            lag = data_prep.calulate_lag(dt, ticker, prices_df, days=days, hours=hours, minutes=minutes)
        else: #tweet not in trading hours
            lag = float('nan')
        lag_array.append(lag)

    assert len(lag_array) == len(tweets_df.index)
    tweets_df['lag'] = lag_array

    if delete_rows_with_nan_lag:
        tweets_df = tweets_df[np.logical_not(np.isnan(tweets_df['lag']))]

    if dump_file_path:
        pickle.dump(tweets_df, open(dump_file_path, 'wb'))

    return tweets_df

def add_label_col_to_df(tweets_df, percentile1, percentile2):
    tweets_df['label'] = lag_to_label(tweets_df['lag'], percentile1, percentile2)
    return tweets_df

def lag_to_label(lags, percentile1, percentile2):
    assert 0 <= percentile1 <= percentile2 <= 100
    p1 = np.percentile(lags, percentile1)
    p2 = np.percentile(lags, percentile2)
    assert p1 <= p2

    label_array = []
    for lag in lags:
        if lag <= p1:
            label_array.append(0) #negative
        elif lag > p2:
            label_array.append(2) #positive
        else:
            label_array.append(1) #neutral

    return label_array