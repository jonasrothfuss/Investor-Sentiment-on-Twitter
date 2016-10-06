import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.cross_validation import KFold
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import matplotlib.pyplot as plt

SENTIMENT_LEXICON_DIR = '../Data/Sentiment_Lexicon/'
OPINION_FINDER_LEXICON_FILE_NAME = 'sent_dict_opinion_finder.tff'


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

def predict_label_from_vader_compound(tweets_df, classification_thershold = 0.45):
    assert 'lag' in tweets_df.columns
    if not 'vader_compound' in tweets_df.columns:
        tweets_df = add_vader_scores_to_df(tweets_df)

    predicted_label_array = []
    profit_array = []
    for score, lag in zip(tweets_df['vader_compound'], tweets_df['lag']):
        if score < -classification_thershold:
            predicted_label = -1.0
        elif score > classification_thershold:
            predicted_label = 1.0
        else:
            predicted_label = 0.0
        predicted_label_array.append(predicted_label)
        profit_array.append(predicted_label*lag)
    return np.mean(profit_array)

def vader_threshold_profit_plot(tweets_df_30min_lag, tweets_df_1h_lag):
    thresholds = np.arange(0.01, 1.0, 0.01)
    profit_array_30min_lag = [predict_label_from_vader_compound(tweets_df_30min_lag, t) for t in thresholds]
    profit_array_1h_lag = [predict_label_from_vader_compound(tweets_df_1h_lag, t) for t in thresholds]
    print(np.max(profit_array_1h_lag))
    plt.style.use('ggplot')
    plt.plot(thresholds, profit_array_30min_lag, linewidth=2.0)
    plt.plot(thresholds, profit_array_1h_lag, linewidth=2.0)
    plt.legend(['30 min lag', '1 hour lag'], loc='upper right', fontsize=12)
    plt.xlabel('classifiaction threshold t', fontsize=14)
    plt.ylabel('estimated profit per trade', fontsize=14)
    plt.show()


    '''
    plt.bar(range(len(values)), values, align='center', color ='grey')
    plt.xticks(range(len(tickers)), tickers)
    plt.xlim((-1, number_of_tickers +2))
    plt.ylabel('percentage of tweets with respective ticker' + '\n')
    '''

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