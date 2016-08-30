
from data_handling import db_handling
from data_handling import parser
from data_handling import data_prep
import train_sent104

if __name__ == '__main__':
    train_sent104.prepare_data()



'''
if __name__ == '__main__':
    # connect to db_collection
    db = connect_twitter_db()
    tweets_db_collection = get_tweets_collection(db)
    prices_collection = get_prices_collection(db)

    theano.config.floatX = 'float32'

    r = RNTN(20, 3)
    b = r.init_vector([2*r.emb_dim, 1])

    e1 = T.fcol('e1')
    e2 = T.fcol('e1')

    emb1 = np.random.normal(size=[20, 1]).astype(theano.config.floatX)
    emb2 = np.random.normal(size=[20, 1]).astype(theano.config.floatX)

    ru = r.create_recursive_unit()
    xpr = ru(e1, e2)
    print(theano.pp(xpr))

    f = theano.function([e1,e2], xpr)

    y = f(emb1, emb2)

    print(y[0].shape)
    print(y)

'''
