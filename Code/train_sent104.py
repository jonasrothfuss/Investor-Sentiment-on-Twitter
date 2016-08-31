from data_handling import db_handling
from data_handling import data_prep
from TreeLSTM import data_utils
#from TreeLSTM import tree_lstm
import numpy as np
from theano import tensor as T
import theano

SEED = 22
NUM_LABELS = 3

EMB_DIM = 300
HIDDEN_DIM = 100


def prepare_data(vocab_file_path):
    #vocabulary
    vocab = data_utils.Vocab()
    vocab.load(vocab_file_path)

    #load training dataset
    s140 = db_handling.load_sentiment_140(train_data=True, cleaned=True)

    #set seed
    np.random.seed(SEED)

    #train and dev split
    train_indexes = np.random.choice(s140.index, int(len(s140.index) * 0.85), False)
    train_df = s140.iloc[train_indexes,]
    dev_indexes = list(set(s140.index) - set(train_indexes))
    dev_df = s140.iloc[dev_indexes,]
    assert len(dev_df.index) + len(train_df.index) == len(s140.index)


    data = {}
    data['train'] = data_prep.build_rnn_trees(train_df['tweet'], train_df['label'], vocab)
    data['dev'] = data_prep.build_rnn_trees(dev_df['tweet'], dev_df['label'], vocab)

    return vocab, data

def train(vocab_file_path):
    vocab, data = prepare_data(vocab_file_path)

    train_set, dev_set= data['train'], data['dev']

    print('train', len(train_set))
    print('dev', len(dev_set))

    num_emb = vocab.size()
    num_labels = NUM_LABELS

    #assert that data is labeled the right way
    for key, dataset in data.items():
        labels = [label for _, label in dataset]
        print(set(labels))
        assert set(labels) <= set([-1, 0, 1])
    print('num emb:', num_emb)
    print('num labels:', num_labels)
