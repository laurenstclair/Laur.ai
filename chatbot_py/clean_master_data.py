from pandas import DataFrame, read_csv
from nltk import pos_tag, word_tokenize
from re import sub

class DataCleaner:
    '''
    Cleans the data and saves as pickle
    
    converting it to a dataframe where each cell is a list of
    tuples in the form (word, word-type)
    '''
    def __init__(self):
        pass

    def clean_data(self, data):
        '''
        @param: a dataframe of lists of tuples
        ---
        returns a dataframe of cleaned data, where each cell represents a
        sentence via a list of words and their type
        '''
        cleaned_data = DataFrame(columns=["Question", "Answer"])
        # For all lines in the data dataframe
        for j in data.iterrows():
            # Appends the cleaned question and response to the cleaned_data array 
            cleaned_data = cleaned_data.append(
                {
                    "Question":self.tokenize_and_tag_line(self.clean_line(j[1][0])), 
                    "Answer":self.tokenize_and_tag_line(self.clean_line(j[1][1]))
                }, 
                ignore_index=True)
        return cleaned_data

    def clean_line(self, line):
        # This line makes all lowercase, and removes anything that isn't a number 
        return sub(r'[^a-z ]', '', str(line).lower())

    def tokenize_and_tag_line(self, line):
        # Tokenizes the words then tags the tokenized words 
        return pos_tag(word_tokenize(line), None)
