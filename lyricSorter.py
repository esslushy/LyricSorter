import pandas as pd
from pandas import DataFrame as df
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, StratifiedShuffleSplit, cross_val_score
from sklearn.naive_bayes import MultinomialNB#Best NB model for this dataset
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import datasets

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
# print("fitting for Naive Bayes")#most effective w/out cross validation
# clf = MultinomialNB( )
# clf.fit(X_train_vectorized, y_train)
# predicted = clf.predict(X_test_vectorized)
# print(str(predicted) + "\t" + "accuracy: " + str(np.mean(predicted == y_test)))
# conf = confusion_matrix(y_test, predicted)#shows number correct and incorrect for each group
# #The y axis is the Beatles then Taylor swift and the x axis is the guesses first being beatles second being taylor swift
# #this means that 1,1 being higher is good and 2,2 being higher is good as it shows more accuracy
# print(conf)

# print("fitting for Naive Bayes with resampled corpus")#2nd most effective
# clf = MultinomialNB( )
# clf.fit(X_resampledRe, y_resampledRe)
# predicted = clf.predict(X_test_vectorized)
# print(str(predicted) + "\t" + "accuracy: " + str(np.mean(predicted == y_test)))

# print("fitting for support vector machine")#3rd most effective
# clf = SVC( )
# clf.fit(X_train_vectorized, y_train)
# predicted = clf.predict(X_test_vectorized)
# print(str(predicted) + "\t" + "accuracy: " + str(np.mean(predicted == y_test)))

# print("fitting for support vector machine with resampled corpus")#least effective
# clf = SVC( )
# clf.fit(X_resampledRe, y_resampledRe)
# predicted = clf.predict(X_test_vectorized)
# print(str(predicted) + "\t" + "accuracy: " + str(np.mean(predicted == y_test)))

print("Using cross-validation with models.")# cross validation makes sure that you get the real accuracy of
#your model by running it through multiple training sessions. If you get similar accuracies you know your model
#is fairly consistent in its accuracy and what its true accuracy is.
lyricsVectorized = vectorizer.transform([r for r in lyrics]).toarray()
stratSplit = StratifiedShuffleSplit(n_splits=5, random_state=randState, test_size=.2) 
print("Fitting for Support Vector Machine and testing with cross validation")
clf = SVC( )
clf.fit(X_train_vectorized, y_train)
predicted = clf.predict(X_test_vectorized) #0.6008697753080454
print(str(predicted) + "\t" + "accuracy: " + str(np.mean(predicted == y_test)))
scores = cross_validate(estimator=clf, X=lyricsVectorized, y=artists, cv=stratSplit, return_train_score=False, n_jobs=-1)
#Will return test score, fit time, and score time
print(scores["test_score"])
#results: [0.58758154 0.58758154 0.58758154 0.58758154 0.58758154]

print("Fitting for Naive Bayes and testing with cross validation")#remains most effective
clf = MultinomialNB( )
clf.fit(X_train_vectorized, y_train)
predicted = clf.predict(X_test_vectorized) # 0.7593621647741
print(str(predicted) + "\t" + "accuracy: " + str(np.mean(predicted == y_test)))
scores = cross_validate(estimator=clf, X=lyricsVectorized, y=artists, cv=stratSplit, return_train_score=False, n_jobs=-1)
#Will return test score, fit time, and score time
print(scores["test_score"])
#results: [0.77240879 0.76105339 0.75984537 0.75791254 0.7554965 ]


#MultinomialNB was the superiour model  as it had a higher accuracy with only a .022 instability. While the 
#Support vector machine had perfect stability, the lack of examples made it have a poorer accuracy according
#to sklearn documentation.