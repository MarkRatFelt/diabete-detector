import os
import time

import numpy as np
# optional
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential


def runNN(X_train_set, Y_train_set, X_test_set, Y_test_set, n_neurons, n_epochs, seed=155,
          history=True, del_files=True, validation_split=0.0, early_stopping=None):
    np.random.seed(seed)
    nn_model = Sequential()  # create model
    nn_model.add(Dense(n_neurons, input_dim=X_train_set.shape[1], activation='relu'))  # hidden layer
    nn_model.add(Dense(1, activation='sigmoid'))  # output layer
    nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model_callbacks = []
    if early_stopping is not None:
        model_callbacks = [early_stopping]
    if history:
        filepath = "nn_weights_%dneurons-{epoch:02d}.hdf5" % n_neurons
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_weights_only=True,
                                     save_best_only=False, mode='max')
        model_callbacks.append(checkpoint)
        output = nn_model.fit(X_train_set, Y_train_set, epochs=n_epochs, verbose=0,
                              batch_size=X_train_set.shape[0], callbacks=model_callbacks,
                              initial_epoch=0, validation_split=validation_split).history
        time.sleep(0.1)  # hack so that files can be opened in subsequent code
        temp_val_model = Sequential()  # create model
        temp_val_model.add(Dense(n_neurons, input_dim=8, activation='relu'))  # hidden layer
        temp_val_model.add(Dense(1, activation='sigmoid'))  # output layer
        temp_val_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        test_over_time = []
        for i in range(len(output['loss'])):
            temp_val_model.load_weights("nn_weights_%dneurons-%02d.hdf5" % (n_neurons, i))
            scores = temp_val_model.evaluate(X_test_set, Y_test_set, verbose=0)
            test_over_time.append(scores)
            if del_files:
                os.remove("nn_weights_%dneurons-%02d.hdf5" % (n_neurons, i))
        test_over_time = np.array(test_over_time)
        output['test_loss'] = [row[0] for row in test_over_time]
        output['test_acc'] = [row[1] for row in test_over_time]
    else:
        model_output = nn_model.fit(X_train_set, Y_train_set, epochs=n_epochs, verbose=0,
                                    batch_size=X_train_set.shape[0], initial_epoch=0, callbacks=model_callbacks,
                                    validation_split=validation_split)
        validation_size = 0
        output = {}
        if validation_split > 0:
            validation_scores = nn_model.evaluate(model_output.validation_data[0],
                                                  model_output.validation_data[1], verbose=0)
            validation_size = model_output.validation_data[0].shape[0]
            output['validation_loss'] = validation_scores[0]
            output['validation_acc'] = validation_scores[1]
        training_size = X_train_set.shape[0] - validation_size
        train_scores = nn_model.evaluate(X_train_set[0:training_size],
                                         Y_train_set[0:training_size], verbose=0)
        test_scores = nn_model.evaluate(X_test_set, Y_test_set, verbose=0)
        output['train_loss'] = train_scores[0]
        output['train_acc'] = train_scores[1]
        output['test_loss'] = test_scores[0]
        output['test_acc'] = test_scores[1]
    return output
