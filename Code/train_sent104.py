from data_handling import db_handling
from data_handling import data_prep
from TreeLSTM import data_utils
from TreeLSTM import tree_lstm
from TreeLSTM import tree_rnn
import matplotlib.pyplot as plt
import numpy as np
from theano import tensor as T
import theano
import os
import pandas

SEED = 22
NUM_LABELS = 3

EMB_DIM = 300
HIDDEN_DIM = 100
TRAIN_SPLIT_PERCENTAGE = 0.85

LEARNING_RATE = 0.01
DEPENDENCY = False

NUM_EPOCHS = 2 #TODO set to 30

GLOVE_DIR = '/home/jonasrothfuss/Documents/treelstm/data'

LABEL_TRANSLATION_DICT = {-1: 2, 0: 0, 1: 1}

class SentimentModel(tree_lstm.ChildSumTreeLSTM):
    def train_step_inner(self, x, tree, y, y_exists = None):
        self._check_input(x, tree)
        if self.labels_on_nonroot_nodes:
            assert y_exists is not None
            return self._train(x, tree[:, :-1], y, y_exists)
        else:
            return self._train(x, tree[:, :-1], y)

    def train_step(self, root_node, label):
        x, tree, labels, labels_exist = \
            tree_rnn.gen_nn_inputs(root_node, max_degree=self.degree,
                                   only_leaves_have_vals=False,
                                   with_labels=True)
        if self.labels_on_nonroot_nodes:
            y = np.zeros((len(labels), self.output_dim), dtype=theano.config.floatX)
            y[np.arange(len(labels)), labels.astype('int32')] = 1
        else:
            y = np.zeros(self.output_dim, dtype=theano.config.floatX)
            y[labels[-1].astype('int32')] = 1

        loss, pred_y = self.train_step_inner(x, tree, y, labels_exist)
        return loss, pred_y

    def loss_fn_multi(self, y, pred_y, y_exists): #overwrites the RSS loss
        return T.sum(T.nnet.categorical_crossentropy(pred_y, y) * y_exists)

    def loss_fn(self, y, pred_y): #overwrites the RSS loss
        return -T.sum(y * T.log(pred_y))

    def loss_fn_multi(self, y, pred_y, y_exists): #overwrites the RSS loss
        return T.sum(T.sum(T.sqr(y - pred_y), axis=1) * y_exists, axis=0)

def get_model(num_emb, output_dim, max_degree):
    return SentimentModel(
        num_emb, EMB_DIM, HIDDEN_DIM, output_dim,
        degree=max_degree, learning_rate=LEARNING_RATE,
        trainable_embeddings=True,
        labels_on_nonroot_nodes=False,
        irregular_tree=DEPENDENCY)

def label_dist(label_col):
    label_dist_dict = {}
    for label in set(label_col):
        label_dist_dict[label] = len(label_col[label_col == label].index)
    return label_dist_dict

def gen_oversampled_train_df(train_df):
    label_dist_dict = label_dist(train_df['label'])
    max_sample_label = max(label_dist_dict, key=lambda k: label_dist_dict[k])
    num_of_samples_goal = label_dist_dict[max_sample_label]
    del label_dist_dict[max_sample_label]

    for label in label_dist_dict.keys():
        original_data_with_label = train_df[train_df['label'] == label]
        samples_of_label = label_dist_dict[label]

        if num_of_samples_goal//samples_of_label > 1:
            for i in range(num_of_samples_goal//samples_of_label - 1):
                train_df = pandas.concat([train_df, original_data_with_label])

        number_of_samples_to_add = num_of_samples_goal % samples_of_label
        indexes_to_add = np.random.choice(original_data_with_label.index, number_of_samples_to_add, False)
        s = original_data_with_label.ix[indexes_to_add]
        train_df = pandas.concat([train_df, s])
        assert all(s == num_of_samples_goal for s in label_dist(train_df['label']).values())
    return train_df


def gen_train_and_dev_split(df, perc_train, oversampling = False):
    train_indexes = np.random.choice(df.index, int(len(df.index) * perc_train), False)
    train_df = df.iloc[train_indexes,]
    dev_indexes = list(set(df.index) - set(train_indexes))
    dev_df = df.iloc[dev_indexes,]
    assert len(dev_df.index) + len(train_df.index) == len(df.index)

    if oversampling:
        train_df = gen_oversampled_train_df(train_df)

    return train_df, dev_df

def prepare_data(vocab_file_path):
    #vocabulary
    vocab = data_utils.Vocab()
    vocab.load(vocab_file_path)
    print('Vocab Size: ', len(vocab.word2idx))

    #load training dataset
    s140df = db_handling.load_sentiment_140(train_data=True, cleaned=True)

    #train and dev split
    train_df, dev_df = gen_train_and_dev_split(s140df, TRAIN_SPLIT_PERCENTAGE, oversampling=True)
    data = {}
    data['train'] = data_prep.build_rnn_trees(train_df['tweet'], train_df['label'], vocab)
    data['dev'] = data_prep.build_rnn_trees(dev_df['tweet'], dev_df['label'], vocab)


    return vocab, data

def train(vocab_file_path):
    #set seed
    np.random.seed(SEED)

    vocab, data = prepare_data(vocab_file_path)

    train_set, dev_set = data['train'], data['dev']

    print('train', len(train_set))
    print('dev', len(dev_set))

    #print(str(train_set[1][0]))

    num_emb = vocab.size()
    num_labels = NUM_LABELS
    max_degree = 2

    #assert that data is labeled the right way
    for key, dataset in data.items():
        labels = [label for _, label in dataset]
        assert set(labels) <= set([-1, 0, 1])
    print('num emb:', num_emb)
    print('num labels:', num_labels)

    model = get_model(num_emb, num_labels, max_degree)

    # initialize model embeddings to glove
    embeddings = model.embeddings.get_value()
    glove_vecs = np.load(os.path.join(GLOVE_DIR, 'glove.npy'))
    glove_words = np.load(os.path.join(GLOVE_DIR, 'words.npy'))
    glove_word2idx = dict((word, i) for i, word in enumerate(glove_words))
    for i, word in enumerate(vocab.words):
        if word in glove_word2idx:
            embeddings[i] = glove_vecs[glove_word2idx[word]]
    glove_vecs, glove_words, glove_word2idx = [], [], []
    model.embeddings.set_value(embeddings)

    #perform training and evaluation steps
    for epoch in range(NUM_EPOCHS):
        print('epoch', epoch)
        avg_loss = train_dataset(model, train_set)
        print('avg loss', avg_loss)
        dev_score = evaluate_dataset(model, dev_set)
        print('dev score', dev_score)

def train_dataset(model, data):
    losses = []
    avg_loss = 0.0
    total_data = len(data)
    for i, (tree, _) in enumerate(data):
        loss, pred_y = model.train_step(tree, None)  # labels will be determined by model
        losses.append(loss)
        avg_loss = avg_loss * (len(losses) - 1) / len(losses) + loss / len(losses)
        print('avg loss %.2f at example %d of %d\r' % (avg_loss, i, total_data))
    return np.mean(losses)

def evaluate_dataset(model, data):
    num_correct = 0
    i = 0
    for tree, label in data:
        pred_y = model.predict(tree)[-1]  # root pred is final row
        num_correct += (LABEL_TRANSLATION_DICT[label] == np.argmax(pred_y))
        i += 1
    return float(num_correct) / len(data)

def dataset_label_histogram(s140dataframe):
    plt.hist(s140dataframe['label'], bins=3)
    plt.show()