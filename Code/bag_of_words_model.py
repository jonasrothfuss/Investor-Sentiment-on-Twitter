import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.cross_validation import KFold




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

def classifier_crosval(tweets_df, clf=MultinomialNB()):
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

        classifier = clf.fit(X_train, Y_train)9
        predicted = classifier.predict(X_test)
        profit = np.mean((predicted-1)*lag_test)
        profit_array.append(profit)
        f1 = f1_score(Y_test, predicted, average='weighted')
        f1_array.append(f1)
        accuracy = np.mean(predicted == Y_test)
        accuracy_array.append(accuracy)
        print('Fold', fold_nr, 'accuracy:', accuracy, 'profit:', profit, 'f1_score:', f1)
        fold_nr += 1
    avg_accuracy = np.mean(accuracy_array)
    avg_profit = np.mean(profit_array)
    avg_f1 = np.mean(f1_array)
    print("Overall Accuracy: ", avg_accuracy, "\nOverall Profit", avg_profit, "\nOverall F1:", avg_f1)
    return avg_accuracy, avg_profit, avg_f1
