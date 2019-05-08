#Import Dependencies
from __future__ import print_function
import IPython
import sys
from music21 import *
import numpy as np
from grammar import *
from qa import *
from preprocess import *
from music_utils import *
from data_utils import *
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K

#Dataset
IPython.display.Audio('./data/30s_seq.mp3')   #Listen to snippet of audio.

X, Y, n_values, indices_values = load_music_utils()   #n_values = 78  unique values in the dataset.
                                                      #indices_values: python dictionary mapping from 0-77 to musical values.
print('shape of X:', X.shape)     #(m, Tx, 78)  =   (60, 30, 78)
print('number of training examples:', X.shape[0])
print('Tx (length of sequence):', X.shape[1])
print('total # of unique values:', n_values)
print('Shape of Y:', Y.shape)     #(Ty, m, 78)  =   (30, 60, 78)

#Building the Model
n_a = 64

#Define Layer Objects as global variables.
reshapor = Reshape((1, 78))   #Produces the output of the given shape.
LSTM_cell = LSTM(n_a, return_state = True)
densor = Dense(n_values, activation = 'softmax')

def djmodel(Tx, n_a, n_values):
  X = Input(shape = (Tx, n_values))
  
  a0 = Input(shape = (n_a, ), name = 'a0')
  c0 = Input(shape = (n_a, ), name = 'c0')
  a = a0
  c = c0
  
  outputs = []
  
  for t in range(Tx):
    x = Lambda(lambda x: X[:, t, :])(X)
    x = reshapor(x)
    a, _, c = LSTM_cell(x, initial_state = [a, c])
    out = densor(a)
    outputs.append(out)
  
  model = Model(inputs = [X, a0, c0], outputs = outputs)
  
  return model
  



