import tensorflow as tf
import numpy as np
import random
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error
np.random.seed(0)


input_size=1
num_steps= 2
num_layers=1
nnkeep_prob=0.8
init_learning_rate = 0.001
init_epoch = 3 #5
features = 2
scaler = MinMaxScaler()
outputLayer =1

batch_size = 64
RNNSize=5
init_learning_rate = 0.001
max_epoch = 30 
hidden_activation = tf.nn.relu
num_hidden =[]
RNNType ="RNN"
trainTestRatio = 20
hiddenLayers = 2



def pre_process():

    stock_data = pd.read_csv('AIG.csv')
    stock_data = stock_data.reindex(index=stock_data.index[::-1])
    stock_data[['Volume', 'Close']] = scaler.fit_transform(stock_data[['Volume', 'Close']])


    seq = [price for tup in stock_data[['Volume', 'Close']].values for price in tup]

    seq = np.array(seq)
    print(seq)

    # split into items of features
    seq = [np.array(seq[i * features: (i + 1) * features])
           for i in range(len(seq) // features)]

    # split into groups of num_steps
    X = np.array([seq[i: i + num_steps] for i in range(len(seq) - num_steps)])

    y = np.array([seq[i + num_steps] for i in range(len(seq) - num_steps)])

    # get only close value
    y = [y[i][1] for i in range(len(y))]

    train_size = int(len(X) * (1.0 - (trainTestRatio/100)))
    train_X, test_X = X[:train_size], X[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]

    return train_X, train_y, test_X, test_y


def generate_batches(train_X,train_y,batch_size):
    num_batches = int(len(train_X)) // batch_size
    if batch_size * num_batches < len(train_X):
        num_batches += 1

    batch_indices = range(num_batches)
    # random.shuffle(batch_indices)
    for j in batch_indices:
        batch_X = train_X[j * batch_size: (j + 1) * batch_size]
        batch_y = train_y[j * batch_size: (j + 1) * batch_size]
        assert set(map(len, batch_X)) == {num_steps}
        yield batch_X, batch_y

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    itemindex = np.where(y_true == 0)
    y_true = np.delete(y_true, itemindex)
    y_pred = np.delete(y_pred, itemindex)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def RMSPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    itemindex = np.where(y_true == 0)
    y_true = np.delete(y_true, itemindex)
    y_pred = np.delete(y_pred, itemindex)
    return np.sqrt(np.mean(np.square(((y_true - y_pred) / y_true)), axis=0))


def train_test():


    train_X, train_y, test_X, test_y = pre_process()

    # Create the graph object
    graph = tf.Graph()
    # Add nodes to the graph
    with graph.as_default():

        tf.set_random_seed(1)

        learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
        inputs = tf.placeholder(tf.float32, [None, num_steps, features], name="inputs")
        targets = tf.placeholder(tf.float32, [None], name="targets")
        keep_prob = tf.placeholder(tf.float32, None, name="keep_prob")

        if RNNType == "RNN":
            cell = tf.contrib.rnn.BasicRNNCell(RNNSize, activation=tf.nn.tanh)
        else:
            cell = tf.contrib.rnn.LSTMCell(RNNSize, state_is_tuple=True, activation=tf.nn.tanh)

        val1, state= tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

        val = tf.transpose(val1, [1, 0, 2])

        last_layer = tf.gather(val, int(val.get_shape()[0]) - 1, name="last_lstm_output")

        if hiddenLayers >0:
            for i in range(hiddenLayers):
                layer_no = i + 1
                # hidden layer
                current_layer = tf.layers.dense(last_layer, units=num_hidden[i], activation=hidden_activation,name="hid_{}".format(layer_no))
                last_layer = current_layer

        prediction= tf.layers.dense(last_layer, units=outputLayer)

        loss = tf.reduce_mean(tf.square(prediction - targets))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        minimize = optimizer.minimize(loss)

    #--------------------training------------------------------------------------------

    with tf.Session(graph=graph) as sess:
        tf.set_random_seed(1)

        tf.global_variables_initializer().run()

        iteration = 1

        learning_rates_to_use = [
            init_learning_rate * (
                    learning_rate_decay ** max(float(i + 1 - init_epoch), 0.0)
            ) for i in range(max_epoch)]

        for epoch_step in range(max_epoch):

            current_lr = learning_rates_to_use[epoch_step]

            for batch_X, batch_y in generate_batches(train_X,train_y,batch_size):

                train_data_feed = {
                    inputs: batch_X,
                    targets: batch_y,
                    learning_rate: current_lr,
                    keep_prob: nnkeep_prob
                }

                train_loss, _ , value= sess.run([loss, minimize,val1], feed_dict=train_data_feed)

        saver = tf.train.Saver()
        saver.save(sess, "checkpoints_stock/stock_pred.ckpt")

    # --------------------testing------------------------------------------------------

    with tf.Session(graph=graph) as sess:
        tf.set_random_seed(1)

        saver.restore(sess, tf.train.latest_checkpoint('checkpoints_stock'))

        test_data_feed = {
            keep_prob: 1.0,
            inputs: test_X,
        }

        test_pred = sess.run(prediction, test_data_feed)


        days = range(len(test_y))

        # plt.plot(days,test_y , label='truth close')
        # plt.plot(days, test_pred, label='pred close')
        # plt.legend(loc='upper left', frameon=False)
        # plt.xlabel("day")
        # plt.ylabel("closing price")
        # # plt.ylim((min(test_y), max(test_y)))
        # plt.grid(ls='--')
        # plt.savefig("Stock price Prediction VS Truth mv.png", format='png', bbox_inches='tight', transparent=False)
        # plt.close()

        test_rmse = sqrt(mean_squared_error(test_y, test_pred))
        test_mae = mean_absolute_error(test_y, test_pred)
        test_mape = mean_absolute_percentage_error(test_y, test_pred)
        test_rmspe = RMSPE(test_y, test_pred)

        obj = {}
        obj['f1'] = None
        obj['precision'] = None
        obj['recall'] = None
        obj['accuracy'] = None
        obj['RMSE'] = test_rmse
        obj['MAE'] = test_mae
        obj['MAPE'] = test_mape
        obj['RMSPE'] = test_rmspe

        return obj

def RunModelStock(netInfo):

    global trainTestRatio,RNNSize,batch_size,init_learning_rate,max_epoch,hidden_activation,RNNType,hiddenLayers,outputLayer,num_hidden

    batch_size = netInfo['batchSize']
    RNNSize=netInfo['rnnNodes']
    init_learning_rate = netInfo['learningRate']
    max_epoch = netInfo['epoch']
    trainTestRatio = netInfo['trainTestRatio']
    num_hidden =[netInfo['hLayer1'],netInfo['hLayer2'],netInfo['hLayer3']]
    RNNType =netInfo['nnType']
    outputLayer = netInfo['outputLayer']
    hiddenLayers = netInfo['hiddenLayerNum']
    hidden_activation = netInfo['activation']
    if hidden_activation == "tanh":
        hidden_activation = tf.nn.tanh
    else:
        hidden_activation = tf.nn.relu

    obj = train_test()
    return obj




    











