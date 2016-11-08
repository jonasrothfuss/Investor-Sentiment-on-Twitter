from data_handling import db_handling
from data_handling import parser
from data_handling import data_prep
from data_handling import tweet
from data_handling import stock_quotes
from data_handling import sentiment
from data_handling import tweets_statistic
from data_handling import perform_analysis
from TreeLSTM import data_utils
import bag_of_words_model
import train_model
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import statsmodels as stm

import numpy as np
import random
import logging
import pickle
import datetime
import pytz
import nltk

VOCAB_FILE_PATH = '../Data/vocab_merged.txt'

def setup_logger(log_file_path):
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=log_file_path,
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

def training():
    #setup logger
    setup_logger("../Data_Clean/result_dump/30min_lag/log.txt")

    #test run
    #train_model.train_on_tweets_collected(VOCAB_FILE_PATH, num_epochs=1, dump_dir='../Data/tweets_collected/dump_test/')
    #print('-------- TEST RUN 1 SUCCESSFUL --------')
    #train_model.train_on_tweets_collected(VOCAB_FILE_PATH, num_epochs=1, dump_dir='../Data/tweets_collected/dump_test/', param_load_file_path='../Data/tweets_collected/dump_test/params.pickle')

    #Training on Tweets collected
    train_model.train_on_tweets_collected(vocab_file_path='../Data_Clean/vocab.txt', num_epochs=3, data_dir='../Data_Clean/tree_dump/1h_lag/',
                                  dump_dir='../Data_Clean/result_dump/1h_lag/')

    #Training on SST
    #train_model.train_on_sst(VOCAB_FILE_PATH, num_epochs=10)


def sentiment():
    tweets_df = pickle.load(open(train_model.TWEETS_COLLECTED_DIR + 'tweets_vader.pickle', 'rb'))

    dt1 = datetime.datetime(2015, 10, 24, 14, 30, 0, 0, pytz.UTC)
    dt2 = datetime.datetime(2016, 5, 24, 14, 31, 0, 0, pytz.UTC)
    ticker = 'GS'
    vocab_file_path = '../Data/vocab_merged_old2.txt'
    model_path = '../Result_Data_Storage/sst_sent140_fusion/params_5epochs.pickle'

    lstm_model = train_model.load_model(vocab_file_path, model_path)
    granger_df = perform_analysis.df_for_granger(tweets_df, ticker, model=lstm_model, vocab=train_model.load_vocab(vocab_file_path))

    pickle.dump(granger_df, open('../Data/granger_df_lstm_mfst.pickle', 'wb'))
    #granger_df = pickle.load(open('../Data/granger_df_lstm.pickle', 'rb'))
    granger_df = granger_df[np.logical_not(granger_df.isnull()['sentiment'])]
    print(granger_df)
    perform_analysis.granger_analysis(granger_df)


def evaluation():
    tweets_df = pickle.load(open('../Data_Clean/all_tweets/tweets_all_lags.pickle', 'rb'))
    for label_appendix in ['min_back']:
        for lag in ['lag_' + str(m) +label_appendix for m in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 120, 180, 240, 300]]:
            tweets_df[lag] = tweets_df[lag] * 10000  # convert to basepoints
            for ticker in ['All']:
                for cluster_by in ['day']:
                    if ticker == 'All':
                        summary = bag_of_words_model.vader_clustered_regression(tweets_df, cluster_by, lag_col=lag)
                    else:
                        summary = bag_of_words_model.vader_clustered_regression(tweets_df, cluster_by, ticker)
                    print(lag, ' Ticker:', ticker, 'Clustered by:', cluster_by)
                    print(summary.tables[1].as_latex_tabular(),'\n')

def database_handling():
    prices_df = db_handling.prices_csv_to_prices_df()
    print(prices_df)
    pickle.dump(prices_df, open('../Data/prices.pickle', 'wb'))

    tweets_df = pickle.load(open(train_model.TWEETS_COLLECTED_DIR + 'tweets_1h_lag.pickle', 'rb'))
    print(tweets_df)

def data_preparation():
    prices_df = pickle.load(open('../Data/prices.pickle', 'rb'))
    tweets_df = pickle.load(open('../Data_Clean/all_tweets/tweets_all_lags.pickle', 'rb'))
    print(tweets_df)

    for m in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 120, 180, 240, 300]:
        tweets_df = data_prep.add_lag_col_to_df(tweets_df, prices_df, minutes=m, lag_col_name = 'lag_'+ str(m) +'min_back', reversed=True)
    pickle.dump(tweets_df, open('../Data_Clean/all_tweets/tweets_all_lags.pickle', 'wb'))

def prepare_clean_data():
    #train and dev dataframes
    tweets_df_30min = pickle.load(open('../Data_Clean/train_dev/tweets_30min_lag.pickle', 'rb'))
    tweets_df_1h = pickle.load(open('../Data_Clean/train_dev/tweets_1h_lag.pickle', 'rb'))

    #test dfs
    tweets_test_df_30min = pickle.load(open('../Data_Clean/test/test_30min_lag.pickle', 'rb'))
    tweets_test_df_1h = pickle.load(open('../Data_Clean/test/test_1h_lag.pickle', 'rb'))

    vocab = train_model.load_vocab('../Data_Clean/vocab.txt')

    #data_prep.dump_data_as_rnn_trees(tweets_df_30min, vocab, '../Data_Clean/tree_dump/30min_lag/')
    #data_prep.dump_data_as_rnn_trees(tweets_df_1h, vocab, '../Data_Clean/tree_dump/1h_lag/')

    rnn_trees_test_30min = data_prep.build_rnn_trees(tweets_test_df_30min['tweet'], tweets_test_df_30min['label'], vocab, lag=tweets_test_df_30min['lag'])
    pickle.dump(rnn_trees_test_30min, open('../Data_Clean/tree_dump/30min_lag/test/test_trees.pickle', 'wb'))

    rnn_trees_test_1h = data_prep.build_rnn_trees(tweets_test_df_1h['tweet'], tweets_test_df_1h['label'], vocab, lag=tweets_test_df_1h['lag'])
    pickle.dump(rnn_trees_test_1h, open('../Data_Clean/tree_dump/1h_lag/test/test_trees.pickle', 'wb'))

if __name__ == '__main__':
    np.random.seed(22)
    random.seed(22)

    #database_handling()
    evaluation()
    #data_preparation()
    #sentiment()
    #training()
    #bag_of_words()



