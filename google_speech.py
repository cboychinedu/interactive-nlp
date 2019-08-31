#!/usr/bin/env python3

# Importing the necessary packages
import nltk
import tflearn
import requests
import re
import socket
import random
import pickle
import json
import numpy as np
import tensorflow as tf
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from subprocess import call
from time import sleep
from gtts import gTTS


# Setting the variable for the stemmer function, Python text
# And loading in the json dataset
stemmer = LancasterStemmer()
with open('model/intents.json') as file:
    data = json.load(file)

# Setting a variable to hold the question, and empty dictionary.
# and creating an empty list to hold the filtered sentence
running = True
remote_server = 'www.google.com'
stop_words = stopwords.words('english')
information = {}
filtered_sentence = []


# Loading in the pickle file that contanis the labels, training, words,
# and output data into memory.
try:
    with open('model/data.pickle', "rb") as f:
        words, labels, training, output = pickle.load(f)

    # creating an empty list to store some values.
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    # Creating a loop that would stem the words in the json dataset, and
    # append them into the list created above.
    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])

        # Creating an if statement to append the word that are not present in,
        # the labels list into the label list.
        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    # Stemming the words and converting them into lowercase alphabets,
    # then setting an if statement to remove the "?" character.
    words = [stemmer.stem(w.lower()) for w in words if w != '?']
    words = sorted(list(set(words)))

    # Sorting the value for the words in labels and