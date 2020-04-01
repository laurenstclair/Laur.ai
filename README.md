# Laur.ai
The goal of this project is to create a chatbot capable of a coherent and realistic 30 turns of conversation. This must be accomplished using proper communication, structure, and planning in order to accurately represent the industry standards.

## Scenario
Laur.ai is a chatbot designed to help people practice and simulate first meet-up conversations. This means that the bot will cover a lot of general topics for people to talk about. Users will start by asking a question about something they are passionate about. If Laur does not know the topic she will respond by saying she does not know, and propose a new idea. If she does know the topic, the conversation will continue until we reach an unknown factor in the conversation/end of Laur’s knowledge.

## Implementation
The backend implementation of Laur.ai was done through python3 and the Natural Language Toolkit (nltk) library.

## How to install
1. Create a virtual environment using python3
```
python3 -m venv venv
```
2. Activate the virtual environment
```
source venv/bin/activate
```
3. Install all dependencies
```
pip install -r requiremnts.txt
python -m spacy download en
python chatbot_py/prepare_laurAI.py
```
4. Run the main file
```
python chatbot_py/laur_ai.py
```
## Class Structure
All files are named according to python naming conventions, all lowercase with underscores signifying new words. Our classes are organized using the following structure. Python files can be found in the folder /chatbot_py and includes the files clean_master_data.py, combine_data_to_master.py, process_transcript.py, and laur_ai.py. 

## laur.ai.py
laur_ai.py is built using the nltk library. The data is run throw a series of steps to create a bag of words associated by comment and response after undergoing lemmatization,
  1. Text data is cleaned by the removal of numbers and conversion to lowercase.
  2. Tokenize and tag words: words are split up from phrases to then be categorized based on the type
  3. Lemmatize words: convert words into their base form
  4. Create a bag of words


## Features

### Simple Chatbot
laur_ai uses a mix of natural language processing and semi-supervised learning to produce responses to a given context from the data that it has been trained on. In this way, our chatbot can respond to a wide variety of topics, but is limited by the quality of data that it is trained on.
If multiple contexts in the training data have the same maximum similarity, the model will randomly select a response to one.

### Autocorrect
laur_ai uses an autocorrect function that will guess the most similar word to a misspelling. The autocorrect feature recognizes nouns via Named Entity Recognition and does not attempt to correct any proper noun.

### Response to Unrecognized words
If the maximum similarity found is below a threshold (defeault is 0.05) then the bot will select a noun in the given context and say that it does not know what it means.
