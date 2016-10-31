import numpy as np
import pandas as pd
from data_handling import db_handling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.cross_validation import KFold
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from data_handling import data_prep
from scipy import stats
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

SENTIMENT_LEXICON_DIR = '../Data/Sentiment_Lexicon/'
OPINION_FINDER_LEXICON_FILE_NAME = 'sent_dict_opinion_finder.tff'

MU_30MIN_LAG = -0.000016842
MU_1H_LAG = 0.0000151362334273
SD_30MIN_LAG = 0.003668388961
SD_1H_LAG = 0.00488821739474


def build_features(text_array):
    count_vect = CountVectorizer()
    counts = count_vect.fit_transform(text_array)
    tfidf_transformer = TfidfTransformer()
    frequencies = tfidf_transformer.fit_transform(counts)
    return frequencies, tfidf_transformer

def train_naive_bayes_classifier(tweets_df):
    X_data, tfidf_transformer = build_features(tweets_df['text'])
    Y_data = tweets_df['label']

    return MultinomialNB().fit(X_data, Y_data)

def classifier_crosval(tweets_df, clf=SVC(kernel='linear')):
    X_data, tfidf_transformer = build_features(tweets_df['tweet'])
    Y_data = tweets_df['label'].as_matrix()
    Y_lag =  tweets_df['lag'].as_matrix()
    #y = np.asanyarray(Y_data)
    folds = KFold(X_data.shape[0], n_folds=5)
    accuracy_array = []
    profit_array = []
    f1_array = []
    fold_nr = 1
    for train_index, test_index in folds:
        X_train, X_test = X_data[train_index], X_data[test_index]
        Y_train, Y_test = Y_data[train_index], Y_data[test_index]
        lag_test = Y_lag[test_index]

        print('train classifier')
        classifier = clf.fit(X_train, Y_train)

        #metrics
        predicted = classifier.predict(X_test)
        profit = np.mean((predicted-1)*lag_test)
        accuracy = np.mean(predicted == Y_test)
        f1 = f1_score(Y_test, predicted, average='weighted')

        #append metrics to array
        f1_array.append(f1)
        profit_array.append(profit)
        accuracy_array.append(accuracy)

        print('Fold', fold_nr, 'accuracy:', accuracy, 'profit:', profit, 'f1_score:', f1)
        fold_nr += 1

    #calulate mean of metrics and print summary
    avg_accuracy = np.mean(accuracy_array)
    avg_profit = np.mean(profit_array)
    avg_f1 = np.mean(f1_array)
    print("Overall Accuracy: ", avg_accuracy, "\nOverall Profit", avg_profit, "\nOverall F1:", avg_f1)
    return avg_accuracy, avg_profit, avg_f1

def load_opinion_finder_sent_lexicon(lexicon_file_path=SENTIMENT_LEXICON_DIR + OPINION_FINDER_LEXICON_FILE_NAME):
    polarity_translation_dict = {'neutral': 0.0, 'positive': 1.0, 'weakneg': -0.5, 'negative': -1.0}

    with open(lexicon_file_path, 'r') as f:
        sent_dict = {}
        for line in f.readlines():
            line_tokens = line.replace('\n', '').split(' ')
            word, polarity = None, None
            for token in line_tokens:
                if '=' in token:
                    attr, val = token.split('=')
                    if attr == 'word1':
                        word = val
                    if attr == 'priorpolarity' and val in polarity_translation_dict.keys():
                        polarity = polarity_translation_dict[val]
            if word and polarity:
                sent_dict[word] = polarity
    return sent_dict

def vader_sentiment_scores(text_array):
    sid = SentimentIntensityAnalyzer()
    assert all([type(t) == type('') for t in text_array])
    vs_dict = {'neg': [], 'neu': [], 'pos': [], 'compound': []}
    for i, text in enumerate(text_array):
        if i % 10000 == 0:
            print(i)
        vs = sid.polarity_scores(text)
        for key, value in vs.items():
            vs_dict[key].append(value)
    return vs_dict

def add_vader_scores_to_df(tweets_df):
    assert 'tweet' in tweets_df.columns
    vs_dict = vader_sentiment_scores(tweets_df['tweet'])
    tweets_df['vader_neg'] = vs_dict['neg']
    tweets_df['vader_neu'] = vs_dict['neu']
    tweets_df['vader_pos'] = vs_dict['pos']
    tweets_df['vader_compound'] = vs_dict['compound']
    return tweets_df

def predict_profit_from_vader_compound(tweets_df, classification_thershold = 0.45, return_number_of_trades = False):
    assert 'lag' in tweets_df.columns
    if not 'vader_compound' in tweets_df.columns:
        tweets_df = add_vader_scores_to_df(tweets_df)

    predicted_label_array = []
    profit_array = []
    for score, lag in zip(tweets_df['vader_compound'], tweets_df['lag']):
        if score < -classification_thershold:
            predicted_label = 1.0
        elif score > classification_thershold:
            predicted_label = -1.0
        else:
            predicted_label = 0.0
        predicted_label_array.append(predicted_label)
        if predicted_label != 0.0:
            profit_array.append(predicted_label*lag)
    number_of_trades = len(profit_array)

    if return_number_of_trades:
        return np.mean(profit_array), number_of_trades
    else:
        return np.mean(profit_array)

def p_val_from_profit(avg_profit, n_trades, mu, sd):
    tt = (avg_profit - mu) / (sd/np.sqrt(float(n_trades)))  # t-statistic for mean
    p_val = stats.t.sf(np.abs(tt), n_trades - 1) * 2
    return p_val

def signif_profit_thresholds(n_trades, mu, sd):
    adj_sd = sd/np.sqrt(float(n_trades))
    return stats.norm.ppf(0.025, loc=mu, scale=adj_sd), stats.norm.ppf(0.975, loc=mu, scale=adj_sd)

def vader_short_term_validation_df(tweets_df_30min_lag, tweets_df_1h_lag):
    thresholds = np.arange(0.01, 0.81, 0.05)

    profit_array_30min_lag, n_trades_30min_lag = zip(*[predict_profit_from_vader_compound(tweets_df_30min_lag, t, True) for t in thresholds])
    profit_array_1h_lag, n_trades_1h_lag = zip(*[predict_profit_from_vader_compound(tweets_df_1h_lag, t, True) for t in thresholds])

    mu_30min = np.mean(tweets_df_30min_lag['lag'])
    sd_30min = np.std(tweets_df_30min_lag['lag'])

    p_val_30min_lag = [p_val_from_profit(avg_profit, n_trades, mu_30min, sd_30min) for avg_profit, n_trades in
                       zip(profit_array_30min_lag, n_trades_30min_lag)]
    p_val_1h_lag = [p_val_from_profit(avg_profit, n_trades, MU_1H_LAG, SD_1H_LAG) for avg_profit, n_trades in
                       zip(profit_array_1h_lag, n_trades_1h_lag)]

    return pd.DataFrame({'profit_30min': profit_array_30min_lag,
                         'n_trades_30min': n_trades_30min_lag,
                         'p_val_30_min': p_val_30min_lag,
                         'profit_1h': profit_array_1h_lag,
                         'n_trades_1h': n_trades_1h_lag,
                         'p_val_1h': p_val_1h_lag}, index=thresholds)

def vader_profit_signif_plot(tweets_df_30min_lag):
    thresholds = np.arange(0.01, 0.81, 0.1)
    profit_array_30min_lag, n_trades_30min_lag = zip(
        *[predict_profit_from_vader_compound(tweets_df_30min_lag, t, True) for t in thresholds])
    upper_sign_bound, lower_sign_bound = zip(*[sample_profit_dist(tweets_df_30min_lag, n) for n in n_trades_30min_lag])  #zip(*[signif_profit_thresholds(n, MU_30MIN_LAG, SD_30MIN_LAG) for n in n_trades_30min_lag])
    plt.style.use('ggplot')
    plt.plot(thresholds, profit_array_30min_lag, linewidth=2.0)
    plt.plot(thresholds, upper_sign_bound, linewidth=1.0, linestyle='--')
    plt.plot(thresholds, lower_sign_bound, linewidth=1.0, linestyle='--')
    plt.legend(['avg. profit per trade', 'significane boundary'], loc='upper right', fontsize=12)
    plt.xlabel('classifiaction threshold t', fontsize=14)
    plt.ylabel('profit per trade', fontsize=14)
    plt.show()

def sample_profit_dist(tweets_df, number_of_trades, n_sim = 50000):
    print(number_of_trades)
    profit_array = []
    for i in range(n_sim):
        sample = tweets_df['lag'].sample(int(number_of_trades))
        pos_neg_border = number_of_trades // 2
        profit = np.mean(sample.iloc[range(pos_neg_border)]) - np.mean(sample.iloc[range(pos_neg_border,number_of_trades-1)])
        profit_array.append(profit)
        print(i, profit)
    return np.percentile(profit_array, 2.5), np.percentile(profit_array, 97.5)

def vader_threshold_profit_plot(tweets_df_30min_lag, tweets_df_1h_lag):
    thresholds = np.arange(0.01, 1.0, 0.01)
    profit_array_30min_lag = [predict_profit_from_vader_compound(tweets_df_30min_lag, t) for t in thresholds]
    profit_array_1h_lag = [predict_profit_from_vader_compound(tweets_df_1h_lag, t) for t in thresholds]
    print(np.max(profit_array_1h_lag))
    plt.style.use('ggplot')
    plt.plot(thresholds, profit_array_30min_lag, linewidth=2.0)
    plt.plot(thresholds, profit_array_1h_lag, linewidth=2.0)
    plt.legend(['30 min lag', '1 hour lag'], loc='upper right', fontsize=12)
    plt.xlabel('classifiaction threshold t', fontsize=14)
    plt.ylabel('estimated profit per trade', fontsize=14)
    plt.show()

def classifier_crossval_on_vader_scores(tweets_df, clf=RandomForestClassifier()):
    X_data = tweets_df.as_matrix(columns=['vader_neg', 'vader_neu', 'vader_pos'])
    Y_data = tweets_df['label'].as_matrix()
    Y_lag =  tweets_df['lag'].as_matrix()
    #y = np.asanyarray(Y_data)
    folds = KFold(X_data.shape[0], n_folds=5)
    accuracy_array = []
    profit_array = []
    f1_array = []
    fold_nr = 1
    for train_index, test_index in folds:
        X_train, X_test = X_data[train_index], X_data[test_index]
        Y_train, Y_test = Y_data[train_index], Y_data[test_index]
        lag_test = Y_lag[test_index]

        print('train classifier')
        classifier = clf.fit(X_train, Y_train)

        #metrics
        predicted = classifier.predict(X_test)
        print(set(predicted))
        profit = np.mean((predicted-1)*lag_test)
        accuracy = np.mean(predicted == Y_test)
        f1 = f1_score(Y_test, predicted, average='weighted')

        #append metrics to array
        f1_array.append(f1)
        profit_array.append(profit)
        accuracy_array.append(accuracy)

        print('Fold', fold_nr, 'accuracy:', accuracy, 'profit:', profit, 'f1_score:', f1)
        fold_nr += 1

    #calulate mean of metrics and print summary
    avg_accuracy = np.mean(accuracy_array)
    avg_profit = np.mean(profit_array)
    avg_f1 = np.mean(f1_array)
    print("Overall Accuracy: ", avg_accuracy, "\nOverall Profit", avg_profit, "\nOverall F1:", avg_f1)
    return avg_accuracy, avg_profit, avg_f1

def bulk_sentiment_twitter(start_dt, end_dt, tweets_df, ticker = None, weighted_by_follower = False):
    assert all([col in tweets_df.columns for col in ['ticker', 'vader_compound', 'created_at']])
    relevant_tweets = data_prep.filter_tweets(tweets_df, start_dt, end_dt, ticker)
    if weighted_by_follower:
        aggregated_sent_score = np.sum(relevant_tweets['vader_compound']*relevant_tweets['follower_count'])/(len(relevant_tweets.index)*np.mean(relevant_tweets['follower_count']))
    else:
        aggregated_sent_score = np.sum(relevant_tweets['vader_compound'])/len(relevant_tweets.index)
    return aggregated_sent_score

def vader_lag_correlation(tweets_w_vader):
    tweets_w_vader = tweets_w_vader[tweets_w_vader['vader_compound'] != 0.0]
    print(stats.pearsonr(tweets_w_vader['lag'], tweets_w_vader['vader_compound']))

def vader_clustered_regression(tweets_df, cluster_by, ticker = None):
    assert 'lag', 'vader_compound' in tweets_df.columns
    if ticker:
        tweets_df = tweets_df[tweets_df['ticker'] == ticker]
    tweets_df = add_clusters_to_tweets_df(tweets_df, cluster_by)
    assert 'cluster' in tweets_df.columns
    lm = smf.ols(formula='lag ~ vader_compound', data=tweets_df).fit(cov_type='cluster',
                            cov_kwds={'groups': tweets_df['cluster']}, use_t=True)

    #for table in lm.summary().tables:
    #    print(table.as_latex_tabular())
    return lm.summary()

def add_clusters_to_tweets_df(tweets_df, cluster_by):
    assert cluster_by in ['day', 'hour', 'month', 'week']
    #delete cluster column if already exists
    if 'cluster' in tweets_df.columns:
        del tweets_df['cluster']

    if cluster_by == 'hour':
        clusters = [str(dt.year) + '-' + str(dt.month)+ '-' + str(dt.day) + '-' + str(dt.hour) for dt in tweets_df['created_at']]
        tweets_df['cluster'] = clusters
    if cluster_by == 'day':
        clusters = [str(dt.year) + '-' + str(dt.month)+ '-' + str(dt.day) for dt in tweets_df['created_at']]
        tweets_df['cluster'] = clusters
    if cluster_by == 'week':
        clusters = [str(dt.year) + '-' + str(dt.week) for dt in tweets_df['created_at']]
        tweets_df['cluster'] = clusters
    if cluster_by == 'month':
        clusters = [str(dt.year) + '-' + str(dt.month) for dt in tweets_df['created_at']]
        tweets_df['cluster'] = clusters
    return tweets_df