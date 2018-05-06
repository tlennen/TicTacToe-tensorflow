from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import pandas as pd


def my_input_fn():
    feature_cols, labels = 0,0
    return feature_cols, labels
CSV_COLUMN_NAMES = ['1', '2','3', '4','5','6','7','8','9', 'Outcome']
y_name = "Outcome"
train = pd.read_csv("TrainingData.csv", names=CSV_COLUMN_NAMES, header=0)
train_x, train_y = train, train.pop(y_name)
train = pd.read_csv("TestingData.csv", names=CSV_COLUMN_NAMES, header=0)
test_x, test_y = train, train.pop(y_name)
batch_size = 2000
train_steps = 500
# Fetch the data
input = tf.placeholder(tf.float32, shape=(1,9))
output = tf.placeholder(tf.int32, shape=1)

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(100).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

# Feature columns describe how to use the input.
for x in range(1,6):
    for y in range(1,6):
        my_feature_columns = []
        for key in range(1,10):
            my_feature_columns.append(tf.feature_column.numeric_column(key=str(key)))
        # Build 2 hidden layer DNN with 10, 10 units respectively.
        classifier = tf.estimator.DNNClassifier(
            feature_columns=my_feature_columns,
            # Two hidden layers of 10 nodes each.
            hidden_units=[x*10,x*10],
            # The model must choose between 3 classes.
            n_classes=3)

        my_input_fn = tf.estimator.inputs.numpy_input_fn(
            x=train_x,
            y=train_y,
            batch_size = 1,
            shuffle = False,
            num_epochs = 1)

        print("Training")
        # Train the Model.
        classifier.train(
            input_fn=lambda: train_input_fn(train_x, train_y, batch_size), steps = train_steps)

        print("TRAINED with ", x*10," neurons, set #", y)

        # Evaluate the model.
        eval_result = classifier.evaluate(
            input_fn=lambda:train_input_fn(test_x, test_y,batch_size), steps = train_steps)

        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
"""
# Generate predictions from the model
expected = ['X', 'O', 'X']
predict_x = {
    '1': [1, 0, 1],
    '2': [1, 0, 1],
    '3': [1, 1, 1],
    '4': [0, 2, 1],
    '5': [0, 0, 1],
    '6': [0, 0, 1],
    '7': [0, 0, 1],
    '8': [0, 0, 1],
    '9': [0, 0, 1]
}

predictions = classifier.predict(
    input_fn=lambda:iris_data.eval_input_fn(predict_x,
                                            labels=None,
                                            batch_size=batch_size))

template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

for pred_dict, expec in zip(predictions, expected):
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print(template.format(iris_data.SPECIES[class_id],
                          100 * probability, expec))

"""