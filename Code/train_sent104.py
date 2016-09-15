from data_handling import db_handling
from data_handling import data_prep
from TreeLSTM import data_utils
from model import SentimentModel
import matplotlib.pyplot as plt
import numpy as np
import sklearn

SEED = 22
NUM_LABELS = 3

EMB_DIM = 300
HIDDEN_DIM = 100
TRAIN_SPLIT_PERCENTAGE = 0.85

LEARNING_RATE = 0.01
DEPENDENCY = False

NUM_EPOCHS = 2 #TODO set to 30

GLOVE_DIR = '/home/jonasrothfuss/Documents/treelstm/data'
PARAMS_PICKLE_FILE_PATH = db_handling.sentiment104_PATH + 'params.pickle'

LABEL_TRANSLATION_DICT = {-1: 2, 0: 0, 1: 1}

def get_model(num_emb, output_dim, max_degree):
    return SentimentModel(
        num_emb, EMB_DIM, HIDDEN_DIM, output_dim,
        degree=max_degree, learning_rate=LEARNING_RATE,
        trainable_embeddings=True,
        labels_on_nonroot_nodes=False,
        irregular_tree=DEPENDENCY)

def train(vocab_file_path):
    #set seed
    np.random.seed(SEED)

    s140df = db_handling.load_sentiment_140(train_data=True, cleaned=True)
    vocab, data = data_prep.prepare_data(vocab_file_path, s140df, TRAIN_SPLIT_PERCENTAGE)

    train_set, dev_set = data['train'], data['dev']

    print('train', len(train_set))
    print('dev', len(dev_set))

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

    # initialize model embeddings with GloVe
    model.initialize_model_embeddings(vocab, GLOVE_DIR)

    #perform training and evaluation steps
    for epoch in range(NUM_EPOCHS):
        print('epoch', epoch)
        avg_loss = train_dataset(model, train_set)
        print('avg loss', avg_loss)
        dev_accuracy, f1_score, _ = evaluate_dataset(model, dev_set)
        print('dev accuracy', dev_accuracy, 'f1 score', f1_score)

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
        transformed_label = LABEL_TRANSLATION_DICT[label]
        predicted_label = np.argmax(pred_y)
        num_correct += (transformed_label == predicted_label)
        conf_matrix[transformed_label, predicted_label] += 1
        label_array.append(transformed_label)
        pred_array.append(predicted_label)
        i += 1

    accuracy = float(num_correct) / len(data)
    f1_score = sklearn.metrics.f1_score(label_array, pred_array)
    return accuracy, f1_score, conf_matrix

def dataset_label_histogram(s140dataframe):
    plt.hist(s140dataframe['label'], bins=3)
    plt.show()