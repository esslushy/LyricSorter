import pandas as pd
from pandas import DataFrame as df
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
#vars
stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()
stopWords = set(stopwords.words("english"))#get list of stopwords
artistName = "Artist_Name"
lyric = "Lyric"
lyricsDataFrame = pd.read_table("lyrics.txt", names=[artistName, lyric])
randState = 7
def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))
vectorizer = CountVectorizer(analyzer=stemmed_words,
                             tokenizer=None,
                             lowercase=True,
                             preprocessor=None,
                             max_features=5000
                             )
#code
print(lyricsDataFrame)
print(lyricsDataFrame.isnull().values.any())#quick check for nulls
print(lyricsDataFrame[artistName].unique())#make sure there are only 2 unique artists and no misspellings
#data is correct
#wrangling and preprocessing
#filter out punctuation, numbers, set everything to lower case
lyricsDataFrame[lyric] = lyricsDataFrame[lyric].apply(lambda x: re.sub("[^a-zA-Z+ +']"," ",x).lower())#remove all things that aren't letter spaces or apostrophes
print(lyricsDataFrame)#remove all none necessary characters
print(lyricsDataFrame[lyric].value_counts())#how many repeats
valueCountDataFrame = pd.DataFrame(lyricsDataFrame[lyric].value_counts().reset_index())
print(valueCountDataFrame)#print repeats in the dataframe in another data frame
#repeats not that effecting and stuff subbbed out correctly
lyricsDataFrame[lyric] = lyricsDataFrame[lyric].apply(lambda x: " ".join([stemmer.stem(word=word) for word in analyzer(x)]))
print(lyricsDataFrame)#makes all words into their basic root such as fish from fishing
lyricsDataFrame[lyric] = lyricsDataFrame[lyric].apply(lambda x: " ".join([w for w in word_tokenize(x) if not w in stopWords]))
print(lyricsDataFrame)#removes stopwords
#train test splits
lyrics = lyricsDataFrame[lyric].values#x values
artists = lyricsDataFrame[artistName].values# y values
X_train, X_test, y_train, y_test = train_test_split(lyrics, artists, test_size=.2, random_state=randState)#test size is out of 1. 20% of the data is test
#vectorize (best to do it after processing and split)
X_train_vectorized = vectorizer.fit_transform([r for r in X_train]).toarray()
X_test_vectorized = vectorizer.transform([r for r in X_test]).toarray()
#resampled corpus
print("Resampling corpus\n")
rs = RandomOverSampler()
X_resampledRe, y_resampledRe = rs.fit_sample(X_train_vectorized, y_train)
#model
print("fitting for Bernoulli Naive Bayes")
clf = BernoulliNB()
clf.fit(X_train_vectorized, y_train)
predicted = clf.predict(X_test_vectorized)
print(str(predicted) + "\t" + "accuracy: " + str(np.mean(predicted == y_test)))

print("fitting for Naive Bayes")
clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)
predicted = clf.predict(X_test_vectorized)
print(str(predicted) + "\t" + "accuracy: " + str(np.mean(predicted == y_test)))

print("fitting for Naive Bayes with resampled corpus")
clf = MultinomialNB()
clf.fit(X_resampledRe, y_resampledRe)
predicted = clf.predict(X_test_vectorized)
print(str(predicted) + "\t" + "accuracy: " + str(np.mean(predicted == y_test)))

print("fitting for support vector machine")
clf = SVC()
clf.fit(X_train_vectorized, y_train)
predicted = clf.predict(X_test_vectorized)
print(str(predicted) + "\t" + "accuracy: " + str(np.mean(predicted == y_test)))

print("fitting for support vector machine with resampled corpus")
clf = SVC()
clf.fit(X_resampledRe, y_resampledRe)
predicted = clf.predict(X_test_vectorized)
print(str(predicted) + "\t" + "accuracy: " + str(np.mean(predicted == y_test)))