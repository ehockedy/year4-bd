import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.snowball import SnowballStemmer


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


def split_data(data, classifications, train_percent, test_percent):
    shuffled_data, shuffled_class = shuffle(data, classifications)  # Randomize the order
    cutoff = int(len(data) * train_percent)
    print(cutoff)
    train = shuffled_data[0:cutoff+1]
    train_class = shuffled_class[0:cutoff+1]
    test = shuffled_data[cutoff:]
    test_class = shuffled_class[cutoff:]
    return train, train_class, test, test_class


def show_frequency_of_words(idx, data_features, vocab):
    for i in range(0, len(data_features[idx])):
        print(data_features[idx][i], vocab[i])


def show_occurence_of_all_words(vocab, dist):
    for i in sorted(zip(vocab, dist), key=lambda x: x[1]):
        print(i)


def classify_nb():
    # Read in data
    data = pd.read_csv("news_ds.csv", header=0)

    # Convert a collection of text documents to a matrix of token counts
    vectorizer = CountVectorizer(analyzer="word",  # Whether words or character n-grams
                                tokenizer=None,
                                preprocessor=None,
                                stop_words=None,
                                max_features=4000)

    # Get important words from each document
    text = get_article_texts(data, stem=True)
    classes = get_article_classifications(data)

    data_train, data_classes = shuffle(text, classes)

    # Generate the feature vectors - arrays that hold the number of times each
    # word, out of all words in all articles, appear in this article
    data_features = vectorizer.fit_transform(data_train)  # Get the term-document matrix
    data_features = data_features.toarray()
    # vocab_train = vectorizer.get_feature_names()  # Map from feature integer indices to feature name
    # dist_train = np.sum(data_features, axis=0)  # Get the total occurence for each word over all articles
    # show_frequency_of_words(0, data_features, vocab_train)
    # show_occurence_of_all_words(vocab_train, dist_train)

    # Split data into training and testing
    cutoff = int(0.8 * len(data_features)) + 1
    train_data_features = data_features[0:cutoff]
    test_data_features = data_features[cutoff:]

    train_classes = data_classes[0:cutoff]
    test_classes = data_classes[cutoff:]

    # Train NB classifier
    nb = MultinomialNB()
    nb.fit(train_data_features, train_classes)

    # Test NB classifier
    predictions = nb.predict(test_data_features)
    count = 0
    for i in range(0, len(predictions)):
        if predictions[i] == test_classes[i]:
            count += 1

    # Pring percentage correct
    print(count/len(predictions))

classify_nb()
