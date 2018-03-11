import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics import precision_recall_fscore_support


def text_to_words(idx, data, stem, stop, lower=True):
    article_1 = data["TEXT"][idx]  # Get the article text

    text = re.sub("[^a-zA-Z]", " ", article_1)  # Remove non-letters
    if lower:
        text = text.lower()  # Convert to lowercase
    words = text.split()  # Split into individual words

    if stop:
        sw = set(stopwords.words("english"))  # Get English stop words
        words = [w for w in words if not w in sw]  # Remove stop words

    if stem:
        stemmer = SnowballStemmer("english")
        stemmed = []
        for w in range(0, len(words)):
            stemmed.append(stemmer.stem(words[w]))
        words = stemmed

    return (" ".join(words))  # Return the words as one string


def get_article_texts(csv_data, stem, stop, lower=True):
    articles = []
    for i in range(0, csv_data["TEXT"].size):
        article_main_words = text_to_words(i, csv_data, stem, stop, lower)
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


def classify_nb(data=None, technique=0, stop=True, stem=True, lower=True, random_state=0):
    # Read in data
    if data is None:
        print("Reading in data")
        data = pd.read_csv("news_ds.csv", header=0)
    log = open("parameter_log_NB.txt", "a")
    technique_name = ""
    if technique == 0:
        # Convert a collection of text documents to a matrix of token counts
        vectorizer = CountVectorizer(analyzer="word",  # Whether words or character n-grams
                                     tokenizer=None,
                                     preprocessor=None,
                                     stop_words=None,
                                     max_features=4000)
        technique_name = "TF 1-gram    "
    elif technique == 1:
        vectorizer = TfidfVectorizer(analyzer="word",
                                     tokenizer=None,
                                     preprocessor=None,
                                     stop_words=None,
                                     max_features=4000)
        technique_name = "TF-IDF 1-gram"                                     
    elif technique == 2:
        vectorizer = CountVectorizer(analyzer="word",  # Whether words or character n-grams
                                     tokenizer=None,
                                     preprocessor=None,
                                     stop_words=None,
                                     max_features=4000,
                                     ngram_range=(2, 2))
        technique_name = "TF 2-gram    "                         
    elif technique == 3:
        vectorizer = TfidfVectorizer(analyzer="word",  # Whether words or character n-grams
                                     tokenizer=None,
                                     preprocessor=None,
                                     stop_words=None,
                                     max_features=4000,
                                     ngram_range=(2, 2))
        technique_name = "TF-IDF 2-gram"
    elif technique == 4:
        vectorizer = CountVectorizer(analyzer="word",  # Whether words or character n-grams
                                     tokenizer=None,
                                     preprocessor=None,
                                     stop_words=None,
                                     max_features=4000,
                                     ngram_range=(4, 4))
        technique_name = "TF 4-gram    "
    elif technique == 5:
        vectorizer = TfidfVectorizer(analyzer="word",  # Whether words or character n-grams
                                     tokenizer=None,
                                     preprocessor=None,
                                     stop_words=None,
                                     max_features=4000,
                                     ngram_range=(4, 4))
        technique_name = "TF-IDF 4-gram"          
    elif technique == 6:
        vectorizer = TfidfVectorizer(analyzer="word",  # Whether words or character n-grams
                                     tokenizer=None,
                                     preprocessor=None,
                                     stop_words=None,
                                     max_features=4000,
                                     ngram_range=(1, 4))
        technique_name = "TF-IDF 1to4-gram"  
    elif technique == 7:
        vectorizer = CountVectorizer(analyzer="word",  # Whether words or character n-grams
                                     tokenizer=None,
                                     preprocessor=None,
                                     stop_words=None,
                                     max_features=4000,
                                     ngram_range=(1, 4))
        technique_name = "TF 1to4-gram    " 

    # Get important words from each document

    print("Cleaning text")
    text = get_article_texts(data, stem, stop, lower)
    classes = get_article_classifications(data)

    data_train, data_classes = shuffle(text, classes, random_state=random_state)

    # Generate the feature vectors - arrays that hold the number of times each
    # word, out of all words in all articles, appear in this article
    print("Generating term-document matrix")
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
    print("Training classifier")
    nb.fit(train_data_features, train_classes)

    # Test NB classifier
    predictions = nb.predict(test_data_features)

    precision, recall, fscore, _ = precision_recall_fscore_support(test_classes, predictions)

    count = 0
    for i in range(0, len(predictions)):
        if predictions[i] == test_classes[i]:
            count += 1

    # Pring percentage correct
    print(count/len(predictions))
    log.write("Technique: " + technique_name + ", Stem:" + str(stem) + ", Stop:" + str(stop) + ", Lower:" + str(lower) + " Accuracy:" + str(count/len(predictions)) + 
              ", Precision:" + str(precision) + ", Recall:" + str(recall) + ", Fscore:" + str(fscore) + "\n")


data = pd.read_csv("news_ds.csv", header=0)
print("Reading in data")
options = [True, False] 
for o in options:  # Stem
    for o2 in options:  # Stop
        for i in range(6, 8):
            classify_nb(data, i, o, o2, True, 0)
            classify_nb(data, i, o, o2, True, 1)
            classify_nb(data, i, o, o2, True, 2)
classify_nb(data, 1, True, True, lower=False, random_state=0)
classify_nb(data, 1, True, True, lower=False, random_state=1)
classify_nb(data, 1, True, True, lower=False, random_state=2)

# TODO
# Try with different stemmer
# Precision, recall, F1
# Run again with random training sets, then find average


# Test:
# -  Porter stemmer
# - 1-4 ngram