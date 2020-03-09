# import pandas
import nltk
import numpy
from pandas import DataFrame, Series, read_csv
from re import sub
from nltk.stem import wordnet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import pos_tag
from sklearn.metrics import pairwise_distances
from nltk import word_tokenize
from nltk.corpus import stopwords

from datetime import datetime

class LaurAI:
    """
        Class for the LaurAI chatbot. 
    """

    def __init__(self, data):
        self.data = data[["comment", "response"]]
        self.cleaned_data = DataFrame(columns=["Question", "Answer"])
        # TODO: can just rename columns to save space
        self.finalText = DataFrame(columns=["Lemmas"])
        self.c = CountVectorizer()
        self.bag = None

    def clean_data(self):
        # For all lines in the data dataframe
        for j in self.data.iterrows():
            # Appends the cleaned question and response to the cleaned_data array 
            self.cleaned_data = self.cleaned_data.append(
                {
                    "Question":self.tokenize_and_tag_line(self.clean_line(j[1][0])), 
                    "Answer":self.tokenize_and_tag_line(self.clean_line(j[1][1]))
                }, 
                ignore_index=True)
        
    def clean_line(self, line):
        # This line makes all lowercase, and removes anything that isn't a number 
        return sub(r'[^a-z ]', '', str(line).lower())

    def tokenize_and_tag_line(self, line):
        # Tokenizes the words then tags the tokenized words 
        return pos_tag(word_tokenize(line), None)
            
    def create_lemma_line(self, input_line):
        # We create the lemmatizer object
        lemma = wordnet.WordNetLemmatizer()
        # This is an array for the current line that we will append values to 
        line = []
        for token, ttype in input_line:
            checks = ["a", "v", "r", "n"]
            if(ttype[0].lower() not in checks):
                ttype = "n"
            line.append(lemma.lemmatize(token, ttype[0].lower()))
        return {"Lemmas": " ".join(line)}

    def create_lemma(self):
        # Creates lemmas for the cleaned data (lemma is the lower )
        lemmas = []
        for j in self.cleaned_data.iterrows():
            # print(j[1][1])
            lemmas.append(self.create_lemma_line(j[1][1]))
        self.finalText = self.finalText.append(lemmas, ignore_index=True)

    def create_bag_of_words(self):
        self.bag = DataFrame(self.c.fit_transform(self.finalText["Lemmas"]).toarray(), columns=self.c.get_feature_names(), index=self.data.index)
    
    def askQuestion(self, question):
        # Removes all "stop words"
        valid_words = []
        for i in question.split():
            if i not in stopwords.words("english"):
                valid_words.append(i)
        
        # Clean the data and get tokenized and tagged data
        valid_sentence = self.tokenize_and_tag_line(self.clean_line(" ".join(valid_words)))

        lemma_line = self.create_lemma_line(valid_sentence)
        # valid_question = c.fit_transform(lemma_line).toarray()
        # print(valid_question)

        # lemma_2 = self.c.transform(lemma_line)
        # print(lemma_line)
        # print(lemma_2)

        # create dataframe of one row initialized to zeros
        # this will represent the lemma
        valid_sentence = DataFrame(0, columns=self.bag.columns, index=[0])
        # set column of 1's for words in lemma line
        for i in lemma_line["Lemmas"].split(' '):
            # if we have not seen the lemma before, columns will be added
            # this will create an exception in the cosine similarity calcualtion
            # therefore if we see this, provide a generic response and exit
            if (valid_sentence.loc[:, i] != 0).any():
                print("I am miss pwesident uwu")
                return
            # if input in data, put i
            valid_sentence.loc[:, i] = 1

        # find cosine similarity
        cosine = 1 - pairwise_distances(self.bag, valid_sentence, metric="cosine")
        # prepare data to be used in series with data's index
        cosine = Series(cosine.reshape(1,-1)[0], index=self.data.index)
        
        # determine index of element with highest similarity
        i = cosine.idxmax()
        print("\n", i, ": ", cosine.loc[i])

        # print question and answer
        answer = self.data.loc[i, "response"]
        print(question)
        print(answer)


time1 = datetime.now()
laurBot = LaurAI(read_csv("data/master_data.csv"))
# First we need to clean the data, so it is all lower case and without special characters or numbers
# We can then tokenize the data, which means splitting it up into words instead of a phrase. We also 
# need to know the type of word

# TODO: optimize clean data
#       Create train() or clean() function to run unfrequently, but otherwise 
#       use cleaned data otherwise
laurBot.clean_data()
# print(laurBot.cleaned_data.head())
time2 = datetime.now()
print("Time to clean data: ", (time2 - time1))

# Then we lematize which means to convert the word into it's base form
laurBot.create_lemma()
print(laurBot.finalText.head())
time1 =datetime.now()
print("Time to create lemma: ", (time1 - time2))

# Now we can start to create the bag of words
laurBot.create_bag_of_words()
# print(laurBot.bag.head())
time2 = datetime.now()
print("Time to bag: ", (time2 - time1))

# Then we can ask a question
# laurBot.askQuestion("What is a funny movie we can watch?")
laurBot.askQuestion("hi")
print("Time to ask Question: ", (datetime.now() - time2))
laurBot.askQuestion("how are you?")
laurBot.askQuestion("you are a robot")