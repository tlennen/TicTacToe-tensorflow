from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd

# This model is built to predict the outcome of a Tic-Tac-Toe game, using previously played games between
# an AI and a random player. The data sets are structured as 9 columns for each part of the board
# and the 10th column is the outcome of the game. See the Data-Maker files for more on how they were built.
# The program parameters can be found below and modified to change the number of hidden units and hidden
# layers in the network. A set number of trials are trained and tested for different models with variable
# hidden units. The trials accuracy is then outputted after each trial.

# The batch size is the amount of data that will be used to train and test the network from the
# given .csv files. For example, if batch_size is 2000, 2000 random pieces of data will be selected
# to train and test the network. train_steps is times trained with each piece of data.

# Start unit and end unit is the range of steps the program is testing for. An example of this is:
# Start_unit = 1, end_unit = 10, step_size = 10. number_of_trials = 5, number_of_layers = 1.
# The output will be models of 10,20,30...100 hidden units with a single layer with 5 trials for
# each type of model.

# Created by Tyler Lennen
# Last updated May 11th, 2018

batch_size = 2000
train_steps = 500  # "epochs"
start_unit = 1
end_unit = 10
step_size = 10
number_of_trials = 5
number_of_layers = 1

# Change the files names below to change the training and testing data sets
CSV_COLUMN_NAMES = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'Outcome']
y_name = "Outcome"
train = pd.read_csv("TrainingData.csv", names=CSV_COLUMN_NAMES, header=0)
train_x, train_y = train, train.pop(y_name)
train = pd.read_csv("TestingData.csv", names=CSV_COLUMN_NAMES, header=0)
test_x, test_y = train, train.pop(y_name)


def train_input_fn(features, labels, batches):
    """An input function for training"""
    # Convert the inputs to a Data set.
    data_set = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    data_set = data_set.shuffle(100).repeat().batch(batches)

    # Return the data_set.
    return data_set


if __name__ == "__main__":
    for x in range(start_unit, end_unit + 1):  # Number of hidden units
        for y in range(1, number_of_trials + 1):  # Trials
            my_feature_columns = []
            hidden_layer = []
            for key in range(1, 10):
                my_feature_columns.append(tf.feature_column.numeric_column(key=str(key)))
            for layer in range(0,number_of_layers):
                hidden_layer.append(x*step_size)
            classifier = tf.estimator.DNNClassifier(
                feature_columns=my_feature_columns,
                hidden_units=hidden_layer,
                # The model must choose between 3 outcomes, 'X', 'O', or 'None'.
                n_classes=3)

            print("Training")
            # Train the Model.
            classifier.train(
                input_fn=lambda: train_input_fn(train_x, train_y, batch_size),
                steps=train_steps)
            print("Trained with ", x*step_size, " hidden units with ", number_of_layers, " hidden layer(s). Trial #", y)

            # Evaluate the model.
            eval_result = classifier.evaluate(
                input_fn=lambda: train_input_fn(test_x, test_y, batch_size),
                steps=train_steps)

            # Accuracy is number of correct predictions/ the total predictions
            print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
