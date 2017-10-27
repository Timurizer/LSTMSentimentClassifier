from __future__ import print_function
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.callbacks import ModelCheckpoint


def lstm_nlp(mcw=7500, review_len=250):
    '''
    :param mcw: amount of most common words among the dataset that are taken into account
    :param review_len: amount of words from each review taken among most frequent
    '''

    batch_size = 32

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=mcw)  # downloading data from imdb

    x_train = sequence.pad_sequences(x_train, maxlen=review_len)
    x_test = sequence.pad_sequences(x_test, maxlen=review_len)

    print('Building model...')
    model = Sequential()
    model.add(Embedding(mcw, 128))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # here's the checkpointer. The model is saved into "weights.hdf5" file.
    checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)

    print('Training...')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=15,
              validation_data=(x_test, y_test),
              callbacks=[checkpointer])

    score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)


lstm_nlp()
