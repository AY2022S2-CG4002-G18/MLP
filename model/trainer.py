from __future__ import print_function
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import coremltools
from IPython.display import display, HTML

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import layers

from constants import LABELS, TIME_PERIODS, STEP_DISTANCE, LABEL, BATCH_SIZE, EPOCHS
from dataset import get_data_frame, show_basic_dataframe_info, visualize_data
from utils import create_segments_and_labels

class Trainer:
    def __init__(self, df):
        self.df = df
        self.df_test = None
        self.df_train = None

        self.le = preprocessing.LabelEncoder()
        
        ## Dataset related attributes
        self.input_shape = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_train_hot = None
        self.y_test = None
        self.y_test_hot = None

        ## MLP model
        self.model = None
        self.training_history = None

    def create_model(self):
        # model = Sequential()
        # model.add(Reshape((TIME_PERIODS, 3), input_shape=(self.input_shape,)))
        # model.add(Dense(100, activation='relu'))
        # model.add(Dense(100, activation='relu'))
        # model.add(Dense(100, activation='relu'))
        # model.add(Flatten())
        # model.add(Dense(self.le.classes_.size, activation='softmax'))
        # model = keras.Sequential([
        #     layers.Dense(units=24, input_shape=[240]),
        #     layers.Dense(units=48, activation='relu'),
        #     layers.Dense(units=96, activation='relu'),
        #     layers.Dense(units=48, activation='relu'),
        #     layers.Dense(units=6, activation='softmax')
        # ])
        model = Sequential()
        model.add(Dense(128, input_dim=self.x_train.shape[1], activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.le.classes_.size, activation='softmax'))
        print(model.summary())
        self.model = model

    def visualize_data(self):
        visualize_data(self.df)

    def train_test_split(self):
        # Define column name of the label vector
        LABEL = 'ActivityEncoded'
        # Transform the labels from String to Integer via LabelEncoder
        # Add a new column to the existing DataFrame with the encoded values
        self.df[LABEL] = self.le.fit_transform(self.df['activity'].values.ravel())
        # Differentiate between test set and training set
        self.df_test = self.df[self.df['user-id'] > 28]
        self.df_train = self.df[self.df['user-id'] <= 28]

    def normalize_train(self):
        # Normalize features for training data set (values between 0 and 1)
        # Surpress warning for next 3 operation
        pd.options.mode.chained_assignment = None  # default='warn'
        self.df_train['x-axis'] = self.df_train['x-axis'] / self.df_train['x-axis'].max()
        self.df_train['y-axis'] = self.df_train['y-axis'] / self.df_train['y-axis'].max()
        self.df_train['z-axis'] = self.df_train['z-axis'] / self.df_train['z-axis'].max()
        # Round numbers
        self.df_train = self.df_train.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4})
    
    def normalize_test(self):
        pd.options.mode.chained_assignment = None  # default='warn'
        self.df_test['x-axis'] = self.df_test['x-axis'] / self.df_test['x-axis'].max()
        self.df_test['y-axis'] = self.df_test['y-axis'] / self.df_test['y-axis'].max()
        self.df_test['z-axis'] = self.df_test['z-axis'] / self.df_test['z-axis'].max()
        # Round numbers
        self.df_test = self.df_test.round({'x-axis': 4, 'y-axis': 4, 'z-axis': 4})

    def initialise_train(self):
        self.x_train, self.y_train = create_segments_and_labels(self.df_train,
                                                    TIME_PERIODS,
                                                    STEP_DISTANCE,
                                                    LABEL)

        num_time_periods, num_sensors = self.x_train.shape[1], self.x_train.shape[2]
        num_classes = self.le.classes_.size
        print(list(self.le.classes_))

        self.input_shape = (num_time_periods*num_sensors)

        self.x_train = self.x_train.reshape(self.x_train.shape[0], self.input_shape)
        print('x_train shape:', self.x_train.shape)
        print('input_shape:', self.input_shape)

        self.x_train = self.x_train.astype('float32')
        self.y_train = self.y_train.astype('float32')

        self.y_train_hot = np_utils.to_categorical(self.y_train, num_classes)

        print('New y_train shape: ', self.y_train_hot.shape)

    def initialise_test(self):
        self.x_test, self.y_test = create_segments_and_labels(self.df_test,
                                                    TIME_PERIODS,
                                                    STEP_DISTANCE,
                                                    LABEL)

        self.x_test = self.x_test.reshape(self.x_test.shape[0], self.input_shape)
        print('x_test shape:', self.x_test.shape)
        print('test input_shape:', self.input_shape)

        self.x_test = self.x_test.astype('float32')
        self.y_test = self.y_test.astype('float32')

        self.y_test_hot = np_utils.to_categorical(self.y_test, self.le.classes_.size)
        print('New y_test shape: ', self.y_test_hot.shape)
    
    def train(self):
        callbacks_list = [
            keras.callbacks.ModelCheckpoint(
                filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
                monitor='val_loss', save_best_only=True),
            keras.callbacks.EarlyStopping(monitor='acc', patience=1)
        ]

        self.model.compile(loss='categorical_crossentropy',
                        optimizer='adam', metrics=['accuracy'])


        # Enable validation to use ModelCheckpoint and EarlyStopping callbacks.
        self.training_history  = self.model.fit(self.x_train,
                            self.y_train_hot,
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            callbacks=callbacks_list,
                            validation_split=0.3,
                            verbose=1)
    
    def visualize_training_result(self):
        plt.figure(figsize=(6, 4))
        plt.plot(self.training_history.history['accuracy'], 'r', label='Accuracy of training data')
        plt.plot(self.training_history.history['val_accuracy'], 'b', label='Accuracy of validation data')
        plt.plot(self.training_history.history['loss'], 'r--', label='Loss of training data')
        plt.plot(self.training_history.history['val_loss'], 'b--', label='Loss of validation data')
        plt.title('Model Accuracy and Loss')
        plt.ylabel('Accuracy and Loss')
        plt.xlabel('Training Epoch')
        plt.ylim(0)
        plt.legend()
        plt.show()

    def show_confusion_matrix(self, validations, predictions):
        matrix = metrics.confusion_matrix(validations, predictions)
        plt.figure(figsize=(6, 4))
        sns.heatmap(matrix,
                    cmap='coolwarm',
                    linecolor='white',
                    linewidths=1,
                    xticklabels=LABELS,
                    yticklabels=LABELS,
                    annot=True,
                    fmt='d')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def visualize_testing_result(self):
        score = self.model.evaluate(self.x_test, self.y_test_hot, verbose=1)
        
        y_pred = self.model.predict(self.x_test)

        print(y_pred.argmax(axis=1))
        y_test_int = self.y_test.astype(int)

        matrix = metrics.confusion_matrix(y_test_int, y_pred.argmax(axis=1))
        print(matrix)

        print('\nAccuracy on test data: %0.2f' % score[1])
        print('Loss on test data: %0.2f\n' % score[0])

        print(score)
    
    def save_weight(self):
        
        print("Saving Weights into a text file for C++!")

    def save_test_df(self):
        print("Saving test data", self.x_test.shape)
        
        np.savetxt('test_data.txt', self.x_test)
        np.savetxt('test_label.txt', self.y_test)
        np.savetxt('test_label_one_hot.txt', self.y_test_hot)
