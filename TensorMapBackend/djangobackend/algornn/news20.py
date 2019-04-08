from collections import Counter
import tensorflow as tf
from sklearn.datasets import fetch_20newsgroups
from string import punctuation
from sklearn.preprocessing import LabelBinarizer
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle



embed_size = 150
batch_size = 64
RNNSize=5
learning_rate = 0.001
epoch = 30 
hidden_activation = tf.nn.relu
num_hidden =[]
RNNType ="RNN"
trainTestRatio = 20
hiddenLayers = 2
outputLayer =3


# without sequence length
def pre_process():

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


def get_batches(x, y, batch_size=100):
    n_batches = len(x) // batch_size
    x, y = x[:n_batches * batch_size], y[:n_batches * batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii + batch_size], y[ii:ii + batch_size]



def train_test():

    features, labels, n_words = pre_process()
    train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=0.2, shuffle=False, random_state=42)

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

        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        saver = tf.train.Saver()

    # ----------------------------batch training-----------------------------------------

    with tf.Session(graph=graph) as sess:
        tf.set_random_seed(1)
        sess.run(tf.global_variables_initializer())
        iteration = 1
        for e in range (epoch):
            for ii, (x, y) in enumerate(get_batches(np.array(train_x),  np.array(train_y), batch_size), 1):

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

        return obj


def RunModel20News(netInfo):

    global RNNSize,batch_size,init_learning_rate,max_epoch,hidden_activation,RNNType,hiddenLayers,outputLayer,num_hidden

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

