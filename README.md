# Laur.ai
The goal of this project is to create a chatbot capable of a coherent 30 turns of conversation. This must be accomplished using proper communication, structure, and planning in order to accurately represent the industry standards.

## Scenario
Laur.ai is a chatbot designed to help people practice and simulate first meet-up conversations. This means that the bot will cover a lot of general topics for people to talk about. Users will start by asking a question about something they are passionate about. If Laur does not know the topic she will respond by saying she does not know, and propose a new idea. If she does know the topic, the conversation will continue until we reach an unknown factor in the conversation/end of Laur’s knowledge.

## Implementation
The backend implementation of Laur.ai was done through python3 and the nltk library.

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
