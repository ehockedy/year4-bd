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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Runs on GPU, I think
# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u

def text_to_words(idx, data, stem):
    article_1 = data["TEXT"][idx]  # Get the article text

    text = re.sub("[^a-zA-Z]", " ", article_1)  # Remove non-letters
    text = text.lower()  # Convert to lowercase
    words = text.split()  # Split into individual words

    sw = set(stopwords.words("english"))  # Get English stop words
    words = [w for w in words if not w in sw]  # Remove stop words

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
        articles.append(article_main_words)
    return articles

def get_article_classifications(csv_data):
    classifications = []
    for i in range(0, csv_data["LABEL"].size):
        classifications.append(csv_data["LABEL"][i])
    return classifications

nlp = spacy.load('en')

data = pd.read_csv("news_ds.csv", header=0)[0:1000]
text = get_article_texts(data, stem=False)  # Probably dont stem because capitals add context
classes = get_article_classifications(data)

#vec = nlp(text[0])
#vec2 = nlp(text[1])

#vocab = [lex.text for lex in nlp.vocab]
#vocab.sort()
#print(len(vocab))

#print(len(vec.vector))


data_vec = []
counter = 0
print("Building word2vec")
for i in text:
    vec = nlp(i)
    data_vec.append(vec.vector)  # Maybe do that normalisation thing here, to avoid having to to add min value later down
    if counter % 100 == 0:
        print(counter, "of", len(text))
    counter += 1

print("\nNormalising")
data_vec, classes = shuffle(data_vec, classes)
counter = 0
for i in range(0, len(data_vec)):
    for j in range(0, len(data_vec[i])):
        data_vec[i][j] += abs(min(data_vec[i]))
        if data_vec[i][j] < 0:
            print(data_vec[i][j])
    if len(data_vec[i]) < len(data_vec[0]):
        print("Too short", len(data_vec[i]), len(data_vec[0]))
        data_vec[i] = [0 for i in range(0, len(data_vec[0]))]
    if i % 100 == 0:
        print(i, "of", len(data_vec))
cutoff = int(len(data_vec)*0.7)
data_train = np.array(data_vec[0:cutoff])
data_test = np.array(data_vec[cutoff:])
class_train = np.array(classes[0:cutoff])
class_test = np.array(classes[cutoff:])

print("Len training data:", len(data_train), "\nLen test data", len(data_test))

len_in = len(data_train[0])
len_out = 1
model = Sequential()
model.add(Embedding(len_in, len_out))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

#print(len(data_train), (data_test), (class_train), (class_test))
batch_size = 32
model.fit(data_train, class_train,
          batch_size=batch_size,
          epochs=5,
          validation_data=(data_test, class_test))
score, acc = model.evaluate(data_test, class_test,
                            batch_size=batch_size)

#vocab = [lex.text for lex in nlp.vocab]
#vocab.sort()
#print(vocab)

# Try this: https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py