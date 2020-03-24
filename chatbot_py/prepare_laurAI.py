from pandas import read_csv
from clean_master_data import DataCleaner

print("Downloading laur.ai")

# make necessary imports
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download("wordnet")
nltk.download("stopwords")

print("Training Laur.AI")

# select subset of data
df = read_csv("data/scraped_data.csv")
df = df.sample(n=20000)
df.to_csv("data/master_data.csv", index=False)

# clean selected subset
data_cleaner = DataCleaner()
cleaned_data = data_cleaner.clean_data(df)
cleaned_data.to_pickle("data/master_data_cleaned.pkl")