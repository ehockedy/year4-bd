import spacy
import pandas as pd
import re
import numpy as np
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM, SimpleRNN
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
import os
import time
import random
from sklearn.metrics import precision_recall_fscore_support

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Runs on GPU, I think
# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u

def text_to_words(idx, data, stop=True, stem=False, nlp=None, lower=True):
    article_1 = data["TEXT"][idx]  # Get the article text

    text = re.sub("[^a-zA-Z]", " ", article_1)  # Remove non-letters
    if lower:
        text = text.lower()  # Convert to lowercase
    words = text.split()  # Split into individual words

    if stop:
        sw = set(stopwords.words("english"))  # Get English stop words
        words = [w for w in words if w not in sw]  # Remove stop words

    #if nlp is not None:
    #    words = [w for w in words if nlp(w).has_vector]
    #words = words[0:1000]  # Limit to 1000 words

    if stem:
        #stemmer = SnowballStemmer("english")
        stemmer = PorterStemmer()
        stemmed = []
        for w in range(0, len(words)):
            stemmed.append(stemmer.stem(words[w]))
        words = stemmed
    return (" ".join(words))  # Return the words as one string


def get_article_texts(csv_data, stop, stem, nlp, lower):
    articles = []
    for i in range(0, csv_data["TEXT"].size):
        article_main_words = text_to_words(i, csv_data, stop, stem, nlp, lower)
        # if len(article_main_words) > 1000:
        #     article_main_words = article_main_words[0:1000]
        # elif len(article_main_words) < 1000:
        #     while len(article_main_words) < 1000:
        #         article_main_words.append("")
        articles.append(article_main_words)
        #print(len(article_main_words))
    #padded = sequence.pad_sequences(articles, 1000, truncating="post")

    return articles


def get_article_classifications(csv_data):
    classifications = []
    for i in range(0, csv_data["LABEL"].size):
        classifications.append(csv_data["LABEL"][i])
    return classifications


def generate_w2v(text, nlp):
    data_vec = []
    counter = 0
    print("Building word2vec")
    # https://spacy.io/usage/vectors-similarity 
    for i in text:
        #vec = nlp(i).vector
        words = i.split(" ")
        word_vecs = nlp(words[0]).vector
        for w in range(1, len(words)):
            wv = nlp(words[w]).vector
            word_vecs += wv
        data_vec.append(word_vecs)
        #data_vec.append(vec.vector)  # Maybe do that normalisation thing here, to avoid having to to add min value later down
        if counter % 10 == 0:
            print(counter, "of", len(text))
        counter += 1
    return data_vec


def normalise_data_vecs(data_vec):
    print("\nNormalising")

    #Make all positive. Keras doesn't like it if not
    for i in range(0, len(data_vec)):
        for j in range(0, len(data_vec[i])):
            data_vec[i][j] += 5  # abs(min(data_vec[i]))  # NEED TO UPDATE FOR NEW WORD VECTORS
            #print(min(data_vec[i]), max(data_vec[i]))
            if data_vec[i][j] < 0:
                print(data_vec[i][j])
        if len(data_vec[i]) < len(data_vec[0]):
            print("Too short", len(data_vec[i]), len(data_vec[0]))
            data_vec[i] = [0 for i in range(0, len(data_vec[0]))]
        if i % 10 == 0:
            print(i, "of", len(data_vec))
    return data_vec


def remove_non_vectors(data_vec, classes):
    to_remove = []
    for i in range(0, len(data_vec)):
        if len(data_vec[i]) <= 1:  # Cant remember if its 0 or 1...
            to_remove.append(i)

    for i in to_remove:
        del data_vec[i]
        del classes[i]

    return data_vec, classes


def split_data(data_vec, classes, train_percentage):
    cutoff = int(len(data_vec)*train_percentage)
    data_train = np.array(data_vec[0:cutoff])
    data_test = np.array(data_vec[cutoff:])
    class_train = np.array(classes[0:cutoff])
    class_test = np.array(classes[cutoff:])

    print("Len training data:", len(data_train), "\nLen test data", len(data_test))

    return data_train, data_test, class_train, class_test


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def generate_embedding_matrix(nlp, tokens):
    #vocab = [lex.text for lex in nlp.vocab]
    #vocab.sort()
    #vocab_size = len(vocab)
    vector_size = 0
    i = 0
    while vector_size == 0:
        vec = nlp(nlp.vocab[i].text)
        if vec.has_vector:
            vector_size = len(vec.vector)
        i += 1
    #print("LEN", len(tokens.items()))
    embedding_matrix = np.zeros(shape=(len(tokens.items())+1, vector_size))
    for word, idx in tokens.items():
        if word in nlp.vocab:
            embedding_matrix[idx] = nlp(word).vector  # dont offset to idx-1, since 0 is reserved
        #else:
        #    print(word, idx)
        #    embedding_matrix[idx] = [0 for j in range(0, vector_size)]
        #embedding_matrix[len(tokens.items())+1] = [0 for i in range(0, vector_size)]
    #print(len(tokens.items()))
    return embedding_matrix


# Split into train and test
def keras_lstm(data_train, data_test, class_train, class_test, embed_mat):
    # Create LSTM
    # https://keras.io/getting-started/sequential-model-guide/
    # maybe for lstm https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html 
    # Looks good http://www.volodenkov.com/post/keras-lstm-sentiment-p2/ 
    # Very good https://www.kaggle.com/lystdo/lstm-with-word2vec-embeddings/code 
    # Also potential http://www.orbifold.net/default/2017/01/10/embedding-and-tokenizer-in-keras/ 
    num_data = len(data_train)
    len_data = len(data_train[0])
    len_vec = 1000
    #len_out = 128
    model = Sequential()  # Sequential model is a linear stack of layers
    model.add(Embedding(len_vec, num_data, weights=[embed_mat], input_length=len_data, trainable=False))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, activation='sigmoid', recurrent_activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    batch_size = 1
    model.fit(data_train, class_train,
              batch_size=batch_size,
              epochs=2,
              validation_data=(data_test, class_test))
    score, acc = model.evaluate(data_test, class_test,
                                batch_size=batch_size)
    print(acc)


def tensorflow_lstm(data_train, data_test, class_train, class_test):
    vec_len = len(data_train[0])

    batch_size = 1

    # Placeholders for any given interation
    in_ph = tf.placeholder(tf.float32, shape=[batch_size, 1, vec_len])  # Shape is (batch_size, sizex, sizey)
    out_ph = tf.placeholder(tf.float32, shape=[batch_size, 1])

    input_vector = tf.unstack(tf.transpose(in_ph, perm=[1, 0, 2]))  # https://github.com/tensorflow/tensorflow/issues/7005

    lstm_size = 24

    # Create 128 lstm cells
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_size, state_is_tuple=True)
    outputs, _ = tf.nn.static_rnn(lstm_cell, input_vector, dtype="float32")

    W = init_weights([lstm_size, 10])
    B = init_weights([10])


    # loss_function
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=out_ph))
    #loss = estimator_spec_for_softmax_classification(logits=prediction, labels=labels, mode=mode)

    prediction = tf.nn.softmax(tf.matmul(outputs[-1], W) + B)

    # optimization
    #learning_rate = 0.001
    #opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # model evaluation
    #correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(out_ph, 1))


    cross_entropy = -tf.reduce_sum(out_ph * tf.log(tf.clip_by_value(prediction, 1e-10, 1.0)))
    optimizer = tf.train.AdamOptimizer()
    minimize = optimizer.minimize(cross_entropy)

    mistakes = tf.not_equal(tf.argmax(out_ph, 1), tf.argmax(prediction, 1))
    error = tf.reduce_mean(tf.cast(mistakes, tf.float32))


    # init_op = tf.global_variables_initializer()
    # sess = tf.Session()
    # sess.run(init_op)

    # no_of_batches = int(len(data_train)/batch_size)
    # epoch = 1
    # for i in range(epoch):
    #     ptr = 0
    #     for j in range(no_of_batches):
    #         inp, out = data_train[ptr:ptr+batch_size], class_train[ptr:ptr + batch_size]
    #         inp = np.array(inp).reshape(batch_size, 1, vec_len)
    #         out = np.array(out).reshape(batch_size, 1)
    #         #inp = np.insert(inp, 0, batch_size)
    #         #out = np.insert(out, 0, batch_size)
    #         #print(inp.shape)
    #         ptr += batch_size
    #         sess.run(minimize, {in_ph: inp, out_ph: out})
    #         print(mistakes, tf.argmax(out_ph, 1), tf.argmax(prediction, 1), prediction, outputs, "\n")
    #     print("Epoch - ", str(i))
    # data_test = np.array(data_test).reshape(len(data_test), 1, vec_len)
    # class_test = np.array(class_test).reshape(len(class_test), 1)
    # incorrect = sess.run(error, {in_ph: data_test, out_ph: class_test})
    # print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
    # sess.close()


    # initialize variables
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        no_of_batches = int(len(data_train)/batch_size)
        epoch = 1
        for i in range(epoch):
            ptr = 0
            for j in range(no_of_batches):
                batch_x = data_train[ptr:ptr+batch_size]
                batch_y = class_train[ptr:ptr+batch_size]

                batch_x = np.array(batch_x).reshape(batch_size, 1, vec_len)
                batch_y = np.array(batch_y).reshape(batch_size, 1)

                sess.run(minimize, feed_dict={in_ph: batch_x, out_ph: batch_y})

                # if j % 10 == 0:
                #     acc = sess.run(error, feed_dict={in_ph: batch_x, out_ph: batch_y})
                #     #los = sess.run(loss, feed_dict={'x': batch_x, 'y': batch_y})
                #     print("Accuracy ", acc)
                #     #print("Loss ", los)
                #     print("__________________")
                ptr += batch_size

        no_of_batches_test = int(len(data_test)/batch_size)
        ptr = 0
        for i in range(0, no_of_batches_test):
            batch_x = data_test[ptr:ptr+batch_size]
            batch_y = class_test[ptr:ptr+batch_size]
            batch_x = np.array(batch_x).reshape(batch_size, 1, vec_len)  # https://stackoverflow.com/questions/40430186/tensorflow-valueerror-cannot-feed-value-of-shape-64-64-3-for-tensor-uplace 
            batch_y = np.array(batch_y).reshape(batch_size, 1)
            # a = tf.Print(prediction, [prediction])
            # b = a+1
            # c = tf.Print(out_ph, [out_ph])
            # d = c + 1
            #prediction.eval()
            #out_ph.eval()
            #print(sess.run(copy.copy(prediction)), sess.run(copy.copy(out_ph)))
            pred = tf.Print(prediction, [prediction], message="PRINTING1")
            outputs = tf.Print(outputs, [outputs], message="PRINTING2")
            pred2 = tf.multiply(pred, pred) 
            #sess.run(prediction)
            #sess.run(outputs)
            #out_ph = tf.Print(out_ph, [out_ph])
            print(batch_y[0])  # , prediction, out_ph)
            print("Testing Accuracy:", sess.run(error, feed_dict={in_ph: batch_x, out_ph: batch_y}))
            ptr += batch_size


def run_nn(data_size, use_all_data, save_embed, load_embed,
           lstm_not_rnn=True, nodes=128, dp=0.0, act_func='sigmoid',
           num_e=10, b_size=24, two_lstm=False, num_tokens=20000,
           stop=True, stem=False, lower=True):
    log = open("parameter_log.txt", "a")

    print("Loading spacy corpus")

    nlp_corpus = "en_core_web_lg"  # "en_vectors_web_lg"
    nlp = spacy.load(nlp_corpus)

    if data_size is None:
        data_size = 5068  # Length of whole data
    print("Loading data")
    data = pd.read_csv("news_ds.csv", header=0, nrows=data_size)
    text = get_article_texts(data, stop, stem, nlp=nlp, lower=lower)  # Probably dont stem because capitals add context
    classes = get_article_classifications(data)
    #text, classes = shuffle(text, classes, random_state=0)
    random.seed(0)
    random.shuffle(classes)
    random.seed(0)
    random.shuffle(text)

    train_percentage = 0.8
    tokenizer = Tokenizer(num_tokens)
    len_dat_tr = int(len(text)*train_percentage)

    if use_all_data:  # Use the words from the whole data set
        tokenizer.fit_on_texts(text)  # Find the most common words
    else:  # Use the words from just the training dataset
        tokenizer.fit_on_texts(text[0:len_dat_tr])

    text_ints = tokenizer.texts_to_sequences(text)  # Convert text to lists of integers, with each integer corresponding to a word
    text_ints = sequence.pad_sequences(text_ints, 1000, truncating='pre')  # Make all length 1000

    print("Creating Embedded matrix")
    if load_embed:
        embed_mat = np.load("embed_mat_all_data_" + str(data_size) + ".npy")
    else:
        embed_mat = generate_embedding_matrix(nlp, tokenizer.word_index)  # Create the matrix that maps the word integers to their vectors

    if save_embed:
        np.save("embed_mat_all_data_"+str(data_size), embed_mat)


    # THIS IS A TEMPORARY FIX, THERE ARE CASES OF THE INDEX OF A WORD TOKEN THING BEING LARGER (BY 1) THAN THE SIZE OF THE
    # EMBEDDED MATRIX. CURRENTLY MAPPED TO ZERO, BUT THIS IS BAD SINCE 0 IS A SPECIFIC MAPPING
    # https://github.com/tflearn/tflearn/issues/260
    # https://github.com/tensorflow/tensorflow/issues/2734 
    # for i in range(0, len(text_ints)):
    #     for j in range(0, len(text_ints[i])):
    #         #print(j, len(embed_mat))
    #         if text_ints[i][j] >= len(embed_mat):
    #             text_ints[i][j] = int(len(embed_mat)-1)
    #             print("TOO BIG")

    data_train, data_test, class_train, class_test = split_data(text_ints, classes, train_percentage)  # Split into training and testing data

    #num_words = len(tokenizer.word_index.items())
    #print((num_words))
    vec_size = 0
    i = 0
    while vec_size == 0:
        vec = nlp(nlp.vocab[i].text)
        if vec.has_vector:
            vec_size = len(vec.vector)
        i += 1
    len_data = 1000

    #print(len(embed_mat))
    num_words = len(embed_mat)

    model = Sequential()  # Sequential model is a linear stack of layers
    model.add(Embedding(num_words, vec_size, weights=[embed_mat], input_length=len_data, trainable=False))

    # Hyper params
    num_nodes = nodes  # 64
    drop_prob = dp  # 0.0
    activ = act_func  #'sigmoid'
    num_epochs = num_e
    batch_size = b_size #181#24  # 181  # 14, 18, 181, 362 are all factors of 5068, 181 is a factor of 1267
    two_lstm_layers = two_lstm

    if lstm_not_rnn:
        if two_lstm_layers:
            model.add(LSTM(num_nodes, dropout=drop_prob, recurrent_dropout=drop_prob, activation=activ, return_sequences=True))
            model.add(LSTM(num_nodes, dropout=drop_prob, recurrent_dropout=drop_prob, activation=activ))
        else:
            model.add(LSTM(num_nodes, dropout=drop_prob, recurrent_dropout=drop_prob, activation=activ))
    else:
        model.add(SimpleRNN(64, dropout=drop_prob, recurrent_dropout=drop_prob, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

    # Loss is a summation of the errors made for each example in training or validation sets
    start_time = time.time()
    model.fit(data_train, class_train,
                batch_size=batch_size,
                epochs=num_epochs)
                # validation_data=(data_test, class_test))
    train_time = time.time()
    score, acc = model.evaluate(data_test, class_test,
                                batch_size=1)  # Use predict to get other measures
    classification_time = time.time()
    # acc2 = 0
    predictions = model.predict(data_test, 1, True)
    # for i in range(0, len(class_test)):
    #     print(predictions[i], class_test[i])
    #     if class_test[i] == round(predictions[i][0]):
    #         acc2 += 1
    # print("Predict acc:", acc2/len(predictions))
    precision, recall, fscore, _ = precision_recall_fscore_support(class_test, predictions.round())

    stemmer_type = "Porter"
    print("Score:", score, "Accuracy:", acc)
    log.write("lstm_not_rnn:" + str(lstm_not_rnn) + ", data_size:" + str(data_size) + ", use_all_data:" + str(use_all_data) +
        ", num_nodes:" + str(num_nodes) + ", drop_prob:" + str(drop_prob) + ", activ:" + activ + ", num_epochs:" + str(num_epochs) +
        ", batch_size:" + str(batch_size) + ", two_lstm_layers:" + str(two_lstm_layers) +
        ", stop:" + str(stop) + ", stem:" + str(stem) + ", train_time:" + str(train_time - start_time) + ", test_time:" + str(classification_time - train_time) +
        ", stemmer_type:" + stemmer_type + 
        ", accuracy:" + str(acc) + ", loss:" + str(score) + ", precision:" + str(precision) + ", recall:" + str(recall) + ", fscore" + str(recall) + "\n")
        # ", nlp_corpus:" + nlp_corpus + ", num_tokens:" + str(num_tokens) + 

# Best Params
# Nodes = 32*
# Epochs = 10
# Batch = 24
# Num layers = ??? 
# Activation = sigmoid
# LSTM = yes
# Stem = False
# Stop = False
# Dropout = 0

#node_vals = [25, 32, 45]
#num_e_vals = [4, 10]
#b_size_vals = [24, 181]
stop_vals = [True, False]
stem_vals = [True, False]

# #for n in node_vals:
# for i in stop_vals:
#     for j in stem_vals:
#         for k in stem_vals:  # Lowercase
#             run_nn(data_size=5068, use_all_data=False, save_embed=False, load_embed=True,lstm_not_rnn=False, nodes=32, num_e=10, b_size=24, stop=i, stem=j, lower=k)

num_layers = [True, False]
dropout = [0, 0.1, 0.2]
num_nodes = [25, 32, 40]
# for t in num_layers:
#     for d in dropout:
#         for nn in num_nodes:

# run_nn(data_size=5068, use_all_data=False, save_embed=False, load_embed=True, nodes=32, dp=0, two_lstm=False, stop=False, num_e=1)
# run_nn(data_size=5068, use_all_data=False, save_embed=False, load_embed=True, nodes=32, dp=0, two_lstm=False, stop=False, num_e=4)
# run_nn(data_size=5068, use_all_data=False, save_embed=False, load_embed=True, nodes=32, dp=0, two_lstm=False, stop=False, num_e=7)
# run_nn(data_size=5068, use_all_data=False, save_embed=False, load_embed=True, nodes=32, dp=0, two_lstm=False, stop=False, num_e=10)
# run_nn(data_size=5068, use_all_data=False, save_embed=False, load_embed=True, nodes=32, dp=0, two_lstm=False, stop=False, b_size=1)

if __name__== "__main__":
    run_nn(data_size=5068, use_all_data=False, save_embed=False, load_embed=False, nodes=32, num_e=7, two_lstm=False,stem=False, stop=False, num_tokens=None, b_size=24)

# Show tokens, ranked in order of frequency
#for word, i in tokenizer.word_index.items():
#    print(word, i) 

#tensorflow_lstm(dtrain, dtest, ctrain, ctest)
#keras_lstm(dtrain, ctrain, dtest, ctest)
#vocab = [lex.text for lex in nlp.vocab]
#vocab.sort()
#print(vocab)

# Try this: https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py



# rnn, all data, train size 400, test size 100, batch 10: 59%
# rnn, train data, train size 400, test size 100, batch 10: 56%
# lstm, all data, train size 400, test size 100, batch 10: 59%
# lstm, train data, train size 400, test size 100, batch 10: 55%
# lstm, train data, train size 800, test size 200, batch 10: 62%
# lstm, train data, train size 800, test size 200, batch 5: 62%

# Things to vary:
# - Number of rnn/lstm layers
# - Size of rnn/lstm layers
# - Activation function
# - Number of training epochs
# - Batch size
# - Stemming, 

#TODO:
# - try bigger vocabulary
# - try more tokens (CURRENTLY WRITTEN CODE)
# - Report

# Observations
# RNN is faster to train than LSTM