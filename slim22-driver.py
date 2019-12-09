
"""
Sample script for importing the slim22 module and executing a training/testing example. 
Results are saved/dumpped in a binary files
CSANN LAB--MSU--msu.edu
Contributors:
Atra Akandeh
Fathi Salem
"""
from __future__ import print_function
from keras.datasets import imdb
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers.embeddings import Embedding
#from keras.layers import LSTM
import pickle
import sys
import config
from slim22 import LSTMs

            
def set_variables(*l):
    if l:
        config.lstm = l[0]
        config.batch_size = int(l[1])
        config.nb_epochs = int(l[2])
        config.eta = float(l[3])
        
def main():    
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = config.top_words)
    
    X_train = sequence.pad_sequences(X_train, maxlen=config.max_review_length)
    X_test = sequence.pad_sequences(X_test, maxlen=config.max_review_length)
    
    model = Sequential()
    model.add(Embedding(config.top_words, config.embedding_vector_length, input_length=config.max_review_length, trainable=False))           
    lstmi = LSTMs(implementation= 1, units=config.hidden_units,
            activation=config.act,
            input_shape=X_train.shape[1:], model=config.lstm)
    model.add(lstmi)
    model.add(Dense(1, activation=config.act))
    adam = Adam(lr=config.eta, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=adam , metrics=['accuracy'])
    model.summary()
    hist = model.fit(X_train, y_train, batch_size=config.batch_size, epochs=config.nb_epochs,
                     verbose=1, validation_data=(X_test, y_test))
    fn = '%s.p' % config.name 
    A = hist.history
    pickle.dump( A , open( fn, "wb" ) )

if __name__ == '__main__':
    set_variables(*sys.argv[1:])
    main()