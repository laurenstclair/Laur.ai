from pandas import DataFrame, read_csv
from nltk import pos_tag, word_tokenize
from re import sub


def clean_data(data):
    cleaned_data = DataFrame(columns=["Question", "Answer"])
    # For all lines in the data dataframe
    for j in data.iterrows():
        # Appends the cleaned question and response to the cleaned_data array
        cleaned_data = cleaned_data.append(
            {
                "Question":tokenize_and_tag_line(clean_line(j[1][0])),
                "Answer":tokenize_and_tag_line(clean_line(j[1][1]))
            },
            ignore_index=True)
    return cleaned_data


def clean_line(line):
    # This line makes all lowercase, and removes anything that isn't a number
    return sub(r'[^a-z ]', '', str(line).lower())


def tokenize_and_tag_line(line):
    # Tokenizes the words then tags the tokenized words
    return pos_tag(word_tokenize(line), None)


df = read_csv("../data/chatterbot_lib.csv")
cleaned_data = clean_data(df)
cleaned_data.to_pickle("../data/chatterbot_lib_cleaned.pkl")
