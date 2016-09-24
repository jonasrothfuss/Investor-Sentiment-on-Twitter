from TreeLSTM import tree_lstm
from TreeLSTM import tree_rnn
import os
import numpy as np
import theano.tensor as T
import theano
import pickle
from theano.compat.python2x import OrderedDict

class SentimentModel(tree_lstm.NaryTreeLSTM):
    def __init__(self, *args, **kwargs):
        self.use_ada_delta = kwargs.pop('ada_delta')
        #call super instructor
        super(SentimentModel, self).__init__(*args, **kwargs)

        print('ADA DELTA:', self.use_ada_delta)
        if self.use_ada_delta:
            # initialize intermediate update storage
            self.gradients_sq = [theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX)) for p in
                                 self.params]
            self.deltas_sq = [theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX)) for p in
                              self.params]

            updates = self.ada_delta(self.loss)

            train_inputs = [self.x, self.tree, self.y]
            if self.labels_on_nonroot_nodes:
                train_inputs.append(self.y_exists)
            self._train = theano.function(train_inputs, [self.loss, self.pred_y], updates=updates)
            self._evaluate = theano.function([self.x, self.tree], self.final_state)
            self._predict = theano.function([self.x, self.tree], self.pred_y)


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

    def initialize_model_embeddings(self, vocab, glove_dir):
        embeddings = self.embeddings.get_value()
        glove_vecs = np.load(os.path.join(glove_dir, 'glove.npy'))
        glove_words = np.load(os.path.join(glove_dir, 'words.npy'))
        glove_word2idx = dict((word, i) for i, word in enumerate(glove_words))
        for i, word in enumerate(vocab.words):
            if word in glove_word2idx:
                embeddings[i] = glove_vecs[glove_word2idx[word]]
        glove_vecs, glove_words, glove_word2idx = [], [], []
        self.embeddings.set_value(embeddings)

    def get_params(self, pickle_file_path=None):
        param_dict = {}
        param_dict['W_i'] = self.W_i.get_value()
        param_dict['U_i'] = self.U_i.get_value()
        param_dict['b_i'] = self.b_i.get_value()
        param_dict['W_f'] = self.W_f.get_value()
        param_dict['U_f'] = self.U_f.get_value()
        param_dict['b_f'] = self.b_f.get_value()
        param_dict['W_o'] = self.W_o.get_value()
        param_dict['U_o'] = self.U_o.get_value()
        param_dict['b_o'] = self.b_o.get_value()
        param_dict['W_u'] = self.W_u.get_value()
        param_dict['U_u'] = self.U_u.get_value()
        param_dict['b_u'] = self.b_u.get_value()
        param_dict['W_out'] = self.W_out.get_value()
        param_dict['b_out'] = self.b_out.get_value()
        param_dict['embeddings'] = self.embeddings.get_value()

        if self.use_ada_delta:
            param_dict['gradients_sq'] = [p.get_value() for p in self.gradients_sq]
            param_dict['deltas_sq'] = [p.get_value() for p in self.deltas_sq]

        assert (not self.use_ada_delta and len(param_dict) == len(self.params)) or \
               (self.use_ada_delta and len(param_dict) == (len(self.params)+2))

        if pickle_file_path:
            pickle.dump(param_dict, open(pickle_file_path, 'wb'))

        return param_dict

    def set_params(self, param_dict=None, pickle_file_path=None):
        assert param_dict or pickle_file_path
        if pickle_file_path and not param_dict:
            param_dict = pickle.load(open(pickle_file_path, 'rb'))
        assert (not self.use_ada_delta and len(param_dict) == len(self.params)) or \
               (self.use_ada_delta and len(param_dict) == (len(self.params)+2))

        self.W_i.set_value(param_dict['W_i'])
        self.U_i.set_value(param_dict['U_i'])
        self.b_i.set_value(param_dict['b_i'])
        self.W_f.set_value(param_dict['W_f'])
        self.U_f.set_value(param_dict['U_f'])
        self.b_f.set_value(param_dict['b_f'])
        self.W_o.set_value(param_dict['W_o'])
        self.U_o.set_value(param_dict['U_o'])
        self.b_o.set_value(param_dict['b_o'])
        self.W_u.set_value(param_dict['W_u'])
        self.U_u.set_value(param_dict['U_u'])
        self.b_u.set_value(param_dict['b_u'])
        self.W_out.set_value(param_dict['W_out'])
        self.b_out.set_value(param_dict['b_out'])
        self.embeddings.set_value(param_dict['embeddings'])

        if self.use_ada_delta:
            assert len(param_dict['gradients_sq']) == len(self.gradients_sq)
            assert len(param_dict['deltas_sq']) == len(self.deltas_sq)
            for grad_sq_ten, delta_sq_ten, grad_sq_val, delta_sq_val in zip(self.gradients_sq, self.deltas_sq, param_dict['gradients_sq'], param_dict['deltas_sq']):
                grad_sq_ten.set_value(grad_sq_val)
                delta_sq_ten.set_value(delta_sq_val)

    def ada_delta(self, loss, rho=0.95, eps=1e-8):
        '''AdaDelta with Gradient Clipping'''
        grad = T.grad(loss, self.params)
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), grad)))
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
        scaling_den = T.maximum(5.0, grad_norm)
        updates = OrderedDict()

        for n, (param, grad, grad_sq_old, delta_sq_old) in enumerate(zip(self.params, grad, self.gradients_sq, self.deltas_sq)):
            grad = T.switch(not_finite, 0.01 * param,
                            grad * (5.0 / scaling_den))
            grad_sq_new = rho*grad_sq_old + (1-rho)*(grad**2)
            delta = (T.sqrt(delta_sq_old+eps)/T.sqrt(grad_sq_new+eps))*grad
            delta_sq_new = rho*delta_sq_old + (1-rho)*delta**2
            updates[param] = param - delta
            updates[grad_sq_old] = grad_sq_new
            updates[delta_sq_old] = delta_sq_new
        return updates
