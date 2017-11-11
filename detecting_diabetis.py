import numpy as np
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.ticker import NullFormatter
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold

import keras
from keras.models import Sequential
from keras.layers import Dense

# optional
from keras.callbacks import ModelCheckpoint

from utils import runNN

# fix random seed for reproducibility
seed = 155
np.random.seed(seed)

# load pima indians dataset

# download directly from website
dataset = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data",
    header=None).values
# import from local directory
# dataset = pd.read_csv("pima-indians-diabetes.data", header=None).values
print(dataset[0])

X_train, X_test, Y_train, Y_test = train_test_split(dataset[:, 0:8], dataset[:, 8],
                                                    test_size=0.25, random_state=87)

print('x_train: ' + str(X_train[0]))
print('x_test: ' + str(X_test[0]))
print('y_train: ' + str(Y_train[0]))
print('y_test: ' + str(Y_test[0]))

np.random.seed(seed)

scaler = StandardScaler()

early_stop_criteria = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0,
                                                    patience=20, verbose=0, mode='auto')

nn_output_scaled = runNN(scaler.fit_transform(X_train), Y_train, scaler.fit_transform(X_test),
                         Y_test, 175, 1000, validation_split=0.2, early_stopping=early_stop_criteria)

print(nn_output_scaled)


fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(range(len(nn_output_scaled['loss'])), nn_output_scaled['loss'], linestyle='-', color='deepskyblue',
         label='Training (Standardised)', lw=2)
ax1.plot(range(len(nn_output_scaled['val_loss'])), nn_output_scaled['val_loss'], linestyle='-', color='mediumpurple',
         label='Validation (Standardised)', lw=2)
ax1.plot(range(len(nn_output_scaled['test_loss'])), nn_output_scaled['test_loss'], linestyle='-', color='lightgreen',
         label='Test (Standardised)', lw=2)
ax2.plot(range(len(nn_output_scaled['acc'])), nn_output_scaled['acc'], linestyle='-', color='deepskyblue',
         label='Training (Standardised)', lw=2)
ax2.plot(range(len(nn_output_scaled['val_acc'])), nn_output_scaled['val_acc'], linestyle='-', color='mediumpurple',
         label='Validation (Standardised)', lw=2)
ax2.plot(range(len(nn_output_scaled['test_acc'])), nn_output_scaled['test_acc'], linestyle='-', color='lightgreen',
         label='Test (Standardised)', lw=2)
leg = ax1.legend(bbox_to_anchor=(0.5, 0.95), loc=2, borderaxespad=0., fontsize=13)
ax1.set_xticklabels('')
ax2.set_xlabel('# Epochs', fontsize=14)
ax1.set_ylabel('Loss', fontsize=14)
ax2.set_ylabel('Accuracy', fontsize=14)
plt.tight_layout()
plt.show()
