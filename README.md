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
pip install -r requirements.txt
```
4. Run the main file
```
python laur.ai.py
```
## Class Structure
All files are named according to python naming conventions, all lower case with underscores signifying new words. Our classes are organized using the following structure.

## laur.ai.py
laur.ai.py is built using the nltk library. The data is run throw a series of steps in order to create a bag of words associated by comment and response after undergoing lemmatization, 
  1. Text data is cleaned by removal of numbers and conversion to lowercase
  2. Tokenize and tage: words are split up from phrases to then be categorized based on type
  3. Lemmatize: convert words into their base form
  4. Create bag of words
