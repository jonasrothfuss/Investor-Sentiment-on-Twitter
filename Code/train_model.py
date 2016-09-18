from data_handling import db_handling
from data_handling import data_prep
from data_handling import tweet
from model import SentimentModel
from TreeLSTM import data_utils

import numpy as np
import sklearn
import pickle
import os
import logging
import datetime

SEED = 22
NUM_LABELS = 3

EMB_DIM = 300
HIDDEN_DIM = 100
TRAIN_SPLIT_PERCENTAGE = 0.85

LEARNING_RATE = 0.01
DEPENDENCY = False

NUM_EPOCHS = 2 #TODO set to 30

GLOVE_DIR = '../Data/Glove/'
PARAMS_PICKLE_FILE_PATH = db_handling.sentiment104_PATH + 'params.pickle'
SENT140_PATH = '../Data/sentiment140/'
TWEETS_COLLECTED_DIR = '../Data/tweets_collected/'
SST_DIR_PATH = '../Data/sst/'
VOCAB_FILE = '../Data/vocab_merged.txt'


def get_model(num_emb, output_dim, max_degree):
    return SentimentModel(
        num_emb, EMB_DIM, HIDDEN_DIM, output_dim,
        degree=max_degree, learning_rate=LEARNING_RATE,
        trainable_embeddings=True,
        labels_on_nonroot_nodes=False,
        irregular_tree=DEPENDENCY)

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

def train(vocab, data, param_initialization = None, param_load_file_path = None, param_dump_file_path = None,
          data_batched = False, metrics_dump_path = None):
    #set seed
    np.random.seed(SEED)

    #setup logger
    setup_logger("../Data/log.txt")
    logging.info(" --- STARTED TRAINING SESSION: " + str(datetime.datetime.now()) + ' ---')

    assert type(data) is dict

    num_emb = vocab.size()
    num_labels = NUM_LABELS
    max_degree = 2

    if data_batched:
        #load and concatenate dev data
        dev_set = data_prep.load_and_concatenate_dumps(data['dev'])
        # assert that data is labeled the right way
        assert set([label for _, label in dev_set]) <= set([0, 1, 2])
        for batch_nr, train_dump_file in enumerate(data['train']):
            train_batch = pickle.load(open(train_dump_file, 'rb'))
            assert set([label for _, label in train_batch]) <= set([0, 1, 2])
            logging.info('Batch ' + str(batch_nr + 1) + ' of ' + str(len(data['train'])) + ' OK')
    else:
        train_set, dev_set = data['train'], data['dev']
        # assert that data is labeled the right way
        for key, dataset in data.items():
            labels = [label for _, label in dataset]
            assert set(labels) <= set([0, 1, 2])
            logging.info('train ' + str(len(train_set)))

    logging.info('dev ' + str(len(dev_set)))
    logging.info('num emb: ' + str(num_emb))
    logging.info('num labels: ' + str(num_labels))

    #TODO add test data and final evaluation

    model = get_model(num_emb, num_labels, max_degree)
    if param_initialization:
        model.set_params(param_initialization)
    elif param_load_file_path:
        model.set_params(pickle_file_path=param_load_file_path)

    # initialize model embeddings with GloVe
    model.initialize_model_embeddings(vocab, GLOVE_DIR)

    metrics_dict = {'avg_loss': [], 'dev_accuracy': [], 'f1_score': [], 'conf_matrix': []}

    #perform training and evaluation steps
    for epoch in range(NUM_EPOCHS):
        logging.info('EPOCH ' + str(epoch))
        if data_batched:
            for batch_nr, train_dump_file in enumerate(data['train']):
                logging.info('Batch ' + str(batch_nr + 1) + ' of ' + str(len(data['train'])))
                train_batch = pickle.load(open(train_dump_file, 'rb'))
                avg_loss = train_dataset(model, train_batch)
                logging.info('avg loss ' + str(avg_loss))
        else:
            avg_loss = train_dataset(model, train_set)
            logging.info('avg loss ' + str(avg_loss))
        dev_accuracy, f1_score, conf_matrix = evaluate_dataset(model, dev_set)
        metrics_dict = add_to_metrics_dict(metrics_dict, avg_loss, dev_accuracy, f1_score, conf_matrix)
        logging.info('dev accuracy ' + str(dev_accuracy) + ' f1 score ' + str(f1_score))

    if metrics_dump_path:
        pickle.dump(metrics_dict, open(metrics_dump_path, 'wb'))

    return model.get_params(param_dump_file_path), metrics_dict

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
    #calculates accuracy and f1 metric
    num_correct = 0
    i = 0
    label_array = []
    pred_array = []
    conf_matrix = np.zeros([model.output_dim, model.output_dim])

    for tree, label in data:
        pred_y = model.predict(tree)[-1]  # root pred is final row
        predicted_label = np.argmax(pred_y)
        num_correct += (label == predicted_label)
        conf_matrix[label, predicted_label] += 1
        label_array.append(label)
        pred_array.append(predicted_label)
        i += 1

    accuracy = float(num_correct) / len(data)
    if len(set(label_array)) > 2:
        f1_score = sklearn.metrics.f1_score(label_array, pred_array, average='weighted')
    else:
        f1_score = sklearn.metrics.f1_score(label_array, pred_array, pos_label=2, average='binary')
    return accuracy, f1_score, conf_matrix

def train_on_sent140(vocab_file_path = VOCAB_FILE, param_initialization = None,
                     dump_dir = '../Data/sentiment140/dump_test/'): #TODO change back to dump/train dir
    vocab = load_vocab(vocab_file_path)

    #pass data as dict with dump file paths (indicated by data_batched=True)
    data = {}
    train_dir = dump_dir + 'train/'
    dev_dir = dump_dir + 'dev/'
    data['train'] = [train_dir + file for file in os.listdir(train_dir)]
    data['dev'] = [dev_dir + file for file in os.listdir(dev_dir)]
    return train(vocab, data, data_batched=True, metrics_dump_path=dump_dir + 'metrics.pickle',
                 param_initialization=param_initialization, param_dump_file_path=dump_dir + 'params.pickle')

def train_on_tweets_collected(vocab_file_path = VOCAB_FILE, param_initialization = None,
                              dump_dir = '../Data/tweets_collected/dump_test/'): # TODO change back to dump/train dir
    vocab = load_vocab(vocab_file_path)

    # pass data as dict with dump file paths (indicated by data_batched=True)
    data = {}
    train_dir = dump_dir + 'train/'
    dev_dir = dump_dir + 'dev/'
    data['train'] = [train_dir + file for file in os.listdir(train_dir)]
    data['dev'] = [dev_dir + file for file in os.listdir(dev_dir)]
    return train(vocab, data, data_batched=True, metrics_dump_path=dump_dir + 'metrics.pickle',
                 param_initialization=param_initialization, param_dump_file_path=dump_dir + 'params.pickle')

def train_on_sst(vocab_file_path = VOCAB_FILE, param_initialization = None):
    vocab = load_vocab(vocab_file_path)

    _, data = pickle.load(open(SST_DIR_PATH + 'sst_data.pickle', 'rb'))
    del data['max_degree']
    return train(vocab, data, param_initialization=param_initialization)

def load_vocab(vocab_file_path):
    vocab = data_utils.Vocab()
    vocab.load(vocab_file_path)
    return vocab

def add_to_metrics_dict(metrics_dict, avg_loss, dev_accuracy, f1_score, conf_matrix):
    metrics_dict['avg_loss'].append(avg_loss)
    metrics_dict['dev_accuracy'].append(dev_accuracy)
    metrics_dict['f1_score'].append(f1_score)
    metrics_dict['conf_matrix'].append(conf_matrix)
    return metrics_dict
