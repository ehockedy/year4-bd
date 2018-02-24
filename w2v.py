import spacy
import pandas as pd
import re
import numpy as np
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
import os
import copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Runs on GPU, I think
# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u

def text_to_words(idx, data, stop=True, stem=False):
    article_1 = data["TEXT"][idx]  # Get the article text

    text = re.sub("[^a-zA-Z]", " ", article_1)  # Remove non-letters
    text = text.lower()  # Convert to lowercase
    words = text.split()  # Split into individual words

    if stop:
        sw = set(stopwords.words("english"))  # Get English stop words
        words = [w for w in words if not w in sw]  # Remove stop words

    words = words[0:1000]  # Limit to 1000 words

    if stem:
        stemmer = SnowballStemmer("english")
        stemmed = []
        for w in range(0, len(words)):
            stemmed.append(stemmer.stem(words[w]))
        words = stemmed
    return (" ".join(words))  # Return the words as one string


def get_article_texts(csv_data, stem):
    articles = []
    for i in range(0, csv_data["TEXT"].size):
        article_main_words = text_to_words(i, csv_data, stem)
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

#vec = nlp(text[0])
#vec2 = nlp(text[1])

#vocab = [lex.text for lex in nlp.vocab]
#vocab.sort()
#print(len(vocab))

#print(len(vec.vector))


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


def split_data(data_vec, classes):
    data_vec, classes = shuffle(data_vec, classes)
    cutoff = int(len(data_vec)*0.8)
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

    embedding_matrix = np.zeros(shape=(len(tokens.items()), vector_size))
    for word, idx in tokens.items():
        if word in nlp.vocab:
            embedding_matrix[idx-1] = nlp(word).vector  # idx starts at 1, so -1 to offset
        else:
            embedding_matrix[idx-1] = [0 for i in range(0, vector_size)]
    #print(len(tokens.items()))
    return embedding_matrix


# Split into train and test
def keras_lstm(data_train, data_test, class_train, class_test, embed_mat):
    # Create LSTM
    # https://keras.io/getting-started/sequential-model-guide/
    # maybe for lstm https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html 
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

print("Loading spacy corpus")
nlp = spacy.load('en')  # _core_web_lg')

print("Loading data")
data = pd.read_csv("news_ds.csv", header=0, nrows=500)
text = get_article_texts(data, stem=False)  # Probably dont stem because capitals add context
classes = get_article_classifications(data)

tokenizer = Tokenizer(50000)
tokenizer.fit_on_texts(text)

#data_vectors = generate_w2v(text, nlp)
#data_vectors, classes = remove_non_vectors(data_vectors, classes)

text = tokenizer.texts_to_sequences(text)
text = sequence.pad_sequences(text, 1000, truncating='post')




embed_mat = generate_embedding_matrix(nlp, tokenizer.word_index)


# THIS IS A TEMPORARY FIX, THERE ARE CASES OF THE INDEX OF A WORD TOKEN THING BEING LARGER (BY 1) THAN THE SIZE OF THE
# EMBEDDED MATRIX. CURRENTLY MAPPED TO ZERO, BUT THIS IS BAD SINCE 0 IS A SPECIFIC MAPPING
# https://github.com/tflearn/tflearn/issues/260
# https://github.com/tensorflow/tensorflow/issues/2734 
for i in range(0, len(text)):
    for j in range(0, len(text[i])):
        #print(j, len(embed_mat))
        if text[i][j] >= len(embed_mat):
            text[i][j] = 0
            print("TOO BIG")

data_train, data_test, class_train, class_test = split_data(text, classes)

num_words = len(tokenizer.word_index.items())
vec_size = 0
i = 0
while vec_size == 0:
    vec = nlp(nlp.vocab[i].text)
    if vec.has_vector:
        vec_size = len(vec.vector)
    i += 1
len_data = 1000

model = Sequential()  # Sequential model is a linear stack of layers
model.add(Embedding(num_words, vec_size, weights=[embed_mat], input_length=len_data, trainable=False))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, activation='sigmoid', recurrent_activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

batch_size = 1
model.fit(data_train, class_train,
            batch_size=batch_size,
            epochs=5,
            validation_data=(data_test, class_test))
score, acc = model.evaluate(data_test, class_test,
                            batch_size=batch_size)



# Show tokens, ranked in order of frequency
#for word, i in tokenizer.word_index.items():
#    print(word, i) 

#tensorflow_lstm(dtrain, dtest, ctrain, ctest)
#keras_lstm(dtrain, ctrain, dtest, ctest)
#vocab = [lex.text for lex in nlp.vocab]
#vocab.sort()
#print(vocab)

# Try this: https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py