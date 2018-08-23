import pandas as pd
from pandas import DataFrame as df
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import StratifiedKFold
stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()
#TODO stop word removal, stemming example here https://github.com/ganigavinaya/Music_classification/blob/master/Main.py, then EDA, and later SVM look up
stopWords = set(stopwords.words("english"))#get list of stopwords
artistName = "Artist_Name"
lyric = "Lyric"
lyricsDataFrame = pd.read_table("lyrics.txt", names=[artistName, lyric])
randState = 7
print(lyricsDataFrame)
print(lyricsDataFrame.isnull().values.any())#quick check for nulls
print(lyricsDataFrame[artistName].unique())#make sure there are only 2 unique artists and no misspellings
#data is correct
#wrangling and preprocessing
#filter out punctuation, numbers, set everything to lower case
lyricsDataFrame[lyric] = lyricsDataFrame[lyric].apply(lambda x: re.sub("[^a-zA-Z+\ +']"," ",x).lower())#remove all things that aren't letter spaces or apostrophes
print(lyricsDataFrame)#remove all none necessary characters
print(lyricsDataFrame[lyric].value_counts())#how many repeats
valueCountDataFrame = pd.DataFrame(lyricsDataFrame[lyric].value_counts().reset_index())
print(valueCountDataFrame)#print repeats in the dataframe in another data frame
#repeats not that effecting and stuff subbbed out correctly
#possible need to do stemming first
lyricsDataFrame[lyric] = lyricsDataFrame[lyric].apply(lambda x: " ".join([stemmer.stem(word=word) for word in analyzer(x)]))
print(lyricsDataFrame)#makes all words into their basic root such as fish from fishing
lyricsDataFrame[lyric] = lyricsDataFrame[lyric].apply(lambda x: " ".join([w for w in word_tokenize(x) if not w in stopWords]))
print(lyricsDataFrame)#removes stopwords
#train test splits
kFolds = StratifiedKFold(n_splits=5, random_state=randState)#kfolds set up
lyrics = lyricsDataFrame[lyric]
artists = lyricsDataFrame[artistName]
for train_index, test_index in kFolds.split(lyrics, artists):
    x_train, x_test = lyrics[train_index], lyrics[test_index]
    y_train, y_test = artists[train_index], lyrics[test_index]#split up with kfolds
    #TODO build model and run and train it
