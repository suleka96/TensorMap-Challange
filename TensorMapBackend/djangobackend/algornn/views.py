from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import JSONParser
from . models import RNNConfig, responseConfig
from . serializers import ConfigSerializer,ResultSerializer
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import tensorflow as tf
import numpy as np
from collections import Counter
from string import punctuation
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
from sklearn.preprocessing import MinMaxScaler
import os
import math
import json
from sklearn.datasets import fetch_20newsgroups
from djangobackend.settings import BASE_DIR
np.random.seed(0)


input_size=1
num_steps= 2
num_layers=1
nnkeep_prob=0.8
init_epoch = 3 
features = 2
scaler = MinMaxScaler()
learning_rate_decay = 0.99
outputLayer =1
embed_size = 150

batch_size = 64
RNNSize=5
init_learning_rate = 0.001
max_epoch = 30 
hidden_activation = tf.nn.relu
num_hidden =[]
RNNType ="RNN"
trainTestRatio = 20
hiddenLayers = 2

@csrf_exempt
def RNNConfigView(request):

    if request.method == 'POST':
        netInfo = JSONParser().parse(request)

        global RNNSize,batch_size,init_learning_rate,max_epoch,hidden_activation,RNNType,hiddenLayers,outputLayer,num_hidden

        batch_size = int(netInfo['batchSize'])
        RNNSize= int(netInfo['rnnNodes'])
        init_learning_rate = float(netInfo['learningRate'])
        max_epoch = int(netInfo['epoch'])
        trainTestRatio = int(netInfo['trainTestRatio'])
        num_hidden =[int(netInfo['hLayer1']),int(netInfo['hLayer2']),int(netInfo['hLayer3'])]
        RNNType =netInfo['nnType']
        outputLayer = int(netInfo['outputLayer'])
        hiddenLayers = int(netInfo['hiddenLayerNum'])
        hidden_activation = netInfo['activation']
        if hidden_activation == "tanh":
            hidden_activation = tf.nn.tanh
        else:
            hidden_activation = tf.nn.relu

        print(RNNSize)
        print(trainTestRatio)
        print("OUTpuut",outputLayer)

        if netInfo['dataset'] == "Stock Price":
           resObj = RunModelStock()
        
        elif netInfo['dataset'] == "20 News Groups":
          resObj =  RunModel20News()   
        
        res_serialized = ResultSerializer(resObj)        
        return  JsonResponse(res_serialized.data, safe=False)

def pre_process_stock():
    file_path = os.path.join(BASE_DIR, 'algornn/AIG.csv')
    stock_data = pd.read_csv(file_path)
    stock_data = stock_data.reindex(index=stock_data.index[::-1])
    stock_data[['Volume', 'Close']] = scaler.fit_transform(stock_data[['Volume', 'Close']])
    seq = [price for tup in stock_data[['Volume', 'Close']].values for price in tup]

    seq = np.array(seq)

    seq = [np.array(seq[i * features: (i + 1) * features])
           for i in range(len(seq) // features)]

    X = np.array([seq[i: i + num_steps] for i in range(len(seq) - num_steps)])

    y = np.array([seq[i + num_steps] for i in range(len(seq) - num_steps)])

    y = [y[i][1] for i in range(len(y))]

    train_size = int(len(X) * (1.0 - (trainTestRatio/100)))
    train_X, test_X = X[:train_size], X[train_size:]
    train_y, test_y = y[:train_size], y[train_size:]

    return train_X, train_y, test_X, test_y

def generate_batches_stock(train_X,train_y,batch_size):
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


def train_test_stock():

    train_X, train_y, test_X, test_y = pre_process_stock()

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

            for batch_X, batch_y in generate_batches_stock(train_X,train_y,batch_size):

                train_data_feed = {
                    inputs: batch_X,
                    targets: batch_y,
                    learning_rate: current_lr,
                    keep_prob: nnkeep_prob
                }

                train_loss, _ , value= sess.run([loss, minimize,val1], feed_dict=train_data_feed)
                
                if iteration % 5 == 0:
                    print("Epoch: {}/{}".format(epoch_step,max_epoch),
                          "Iteration: {}".format(iteration),
                          "Train loss: {:.3f}".format(train_loss))
                iteration += 1

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
        test_pred=test_pred.flatten().tolist()

        days = range(len(test_y))

        test_rmse = math.sqrt(mean_squared_error(test_y, test_pred))
        test_mae = mean_absolute_error(test_y, test_pred)
        test_mape = mean_absolute_percentage_error(test_y, test_pred)
        test_rmspe = RMSPE(test_y, test_pred)

        print("RMSE: ", test_rmse)
        print("MAE:", test_mae)
        print("MAPE:", test_mape)
        print("RMSPE:", test_rmspe)

        obj = {}
        obj['f1'] = None
        obj['precision'] = None
        obj['recall'] = None
        obj['accuracy'] = None
        obj['RMSE'] = test_rmse
        obj['MAE'] = test_mae
        obj['MAPE'] = test_mape
        obj['RMSPE'] = test_rmspe
        obj['prediction'] = json.dumps(test_pred)
        obj['true']= json.dumps(test_y)

        return obj

def RunModelStock():
    obj = train_test_stock()
    return obj


def pre_process_news():

    categories_rec = ['rec.sport.baseball', 'rec.sport.hockey']

    rec = fetch_20newsgroups(subset='all', categories=categories_rec, remove=('headers', 'footers', 'quotes'))

    categories_politics = ['talk.politics.mideast']

    politics = fetch_20newsgroups(subset='all', categories=categories_politics, remove=('headers', 'footers', 'quotes'))

    categories_religion = ['soc.religion.christian']

    religion = fetch_20newsgroups(subset='all', categories=categories_religion, remove=('headers', 'footers', 'quotes'))

    data_labels = []

    for post in rec.data:
        data_labels.append(0)

    for post in politics.data:
        data_labels.append(1)

    for post in religion.data:
        data_labels.append(2)

    news_data = []

    for post in rec.data:
        news_data.append(post)

    for post in politics.data:
        news_data.append(post)

    for post in religion.data:
        news_data.append(post)

    newsgroups_data, newsgroups_labels = shuffle(news_data, data_labels, random_state=42)

    words = []
    temp_post_text = []

    for post in newsgroups_data:

        all_text = ''.join([text for text in post if text not in punctuation])
        all_text = all_text.split('\n')
        all_text = ''.join(all_text)
        temp_text = all_text.split(" ")

        for word in temp_text:
            if word.isalpha():
                temp_text[temp_text.index(word)] = word.lower()

        temp_text = list(filter(None, temp_text))
        temp_text = ' '.join([i for i in temp_text if not i.isdigit()])
        words += temp_text.split(" ")
        temp_post_text.append(temp_text)

    dictionary = Counter(words)
    sorted_split_words = sorted(dictionary, key=dictionary.get, reverse=True)
    vocab_to_int = {c: i for i, c in enumerate(sorted_split_words,1)}

    message_ints = []
    for message in temp_post_text:
        temp_message = message.split(" ")
        message_ints.append([vocab_to_int[i] for i in temp_message])


    seq_length = 150
    num_messages = len(temp_post_text)
    features = np.zeros([num_messages, seq_length], dtype=int)
    for i, row in enumerate(message_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    lb = LabelBinarizer()
    labels = lb.fit_transform(newsgroups_labels)


    return features, labels, len(sorted_split_words)+1


def get_batches_news(x, y, batch_size=100):
    n_batches = len(x) // batch_size
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii + batch_size], y[ii:ii + batch_size]



def train_test_news():

    features, labels, n_words = pre_process_news()
    tt_ratio = (trainTestRatio/100)
    train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=tt_ratio, shuffle=False, random_state=42)

    # --------------placeholders-------------------------------------

    # Create the graph object
    graph = tf.Graph()
    # Add nodes to the graph
    with graph.as_default():

        tf.set_random_seed(1)

        inputs_ = tf.placeholder(tf.int32, [None, None], name="inputs")
        labels_ = tf.placeholder(tf.float32, [None, None], name="labels")
        

        # generating random values from a uniform distribution (minval included and maxval excluded)
        embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1), trainable=True)
        embed = tf.nn.embedding_lookup(embedding, inputs_)

        if RNNType == "RNN":
            cell = tf.contrib.rnn.BasicRNNCell(RNNSize,activation=tf.nn.tanh)
        else:
            cell = tf.contrib.rnn.LSTMCell(RNNSize, state_is_tuple=True, activation=tf.nn.tanh)


        outputs, final_state = tf.nn.dynamic_rnn(cell, embed,dtype=tf.float32 )

        last_layer = outputs[:,-1]

        if hiddenLayers >0:

            for i in range(hiddenLayers):
                layer_no = i + 1
                # hidden layer
                current_layer = tf.layers.dense(last_layer, units=num_hidden[i], activation=hidden_activation,name="hid_{}".format(layer_no))
                last_layer = current_layer


        logit = tf.contrib.layers.fully_connected(last_layer, num_outputs=outputLayer, activation_fn=None)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=labels_))

        optimizer = tf.train.AdamOptimizer(init_learning_rate).minimize(cost)

        saver = tf.train.Saver()

    # ----------------------------batch training-----------------------------------------

    with tf.Session(graph=graph) as sess:
        tf.set_random_seed(1)
        sess.run(tf.global_variables_initializer())
        iteration = 1
        for e in range (max_epoch):
            for ii, (x, y) in enumerate(get_batches_news(np.array(train_x),  np.array(train_y), batch_size), 1):

                feed = {inputs_: x,
                        labels_: y,
                        }

                loss, states, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)

                iteration += 1
        saver.save(sess, "checkpoints/sentiment.ckpt")

     # -----------------testing test set-----------------------------------------
        print("starting testing set")
        argmax_pred_array = []
        argmax_label_array = []
        with tf.Session(graph=graph) as sess:
                tf.set_random_seed(1)
                saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
                feed = {inputs_: np.array(test_x)}
                predictions = tf.nn.softmax(logit).eval(feed_dict=feed)

                for i in range(len(predictions)):
                    argmax_pred_array.append(np.argmax(predictions[i], 0))
                    argmax_label_array.append(np.argmax(test_y[i], 0))

                argmax_label_array=(np.array(argmax_label_array,dtype=np.int32)).tolist()
                argmax_pred_array= (np.array(argmax_pred_array,dtype=np.int32)).tolist()

                accuracy = accuracy_score(argmax_label_array, argmax_pred_array)

                f1 = f1_score(argmax_label_array, argmax_pred_array, average="macro")

                recall = recall_score(y_true=argmax_label_array, y_pred=argmax_pred_array, average='macro')

                precision = precision_score(argmax_label_array, argmax_pred_array, average='macro')


        obj = {}
        obj['f1'] = f1
        obj['precision'] = precision
        obj['recall'] = recall
        obj['accuracy'] = accuracy
        obj['RMSE'] = None
        obj['MAE'] = None
        obj['MAPE'] = None
        obj['RMSPE'] = None
        obj['prediction'] = json.dumps(argmax_pred_array)
        obj['true'] = json.dumps(argmax_label_array)

        print(len(argmax_label_array))
        print(len(json.dumps(argmax_label_array)))
        print(json.dumps(argmax_label_array))
        print(argmax_label_array)

        return obj

def RunModel20News():
    obj = train_test_news()
    return obj
