import numpy as np
import os
import tensorflow as tf
from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn
from tensorflow.python.estimator.export.export_output import PredictOutput

INPUT_TENSOR_NAME = "inputs"
SIGNATURE_NAME = "serving_default"
LEARNING_RATE = 0.001


def model_fn(features, labels, mode, params):
    # Connect the first hidden layer to input layer
    # (features["x"]) with relu activation
    
    layer1 = tf.layers.dense(tf.cast(features[INPUT_TENSOR_NAME], tf.float32), 2048, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(seed = 1))
    layer2 = tf.layers.dense(layer1, 1024, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(seed = 1))
    layer3 = tf.layers.dense(layer2, 512, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(seed = 1))
    layer4 = tf.layers.dense(layer3, 256, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(seed = 1))
    layer5 = tf.layers.dense(layer4, 128, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(seed = 1))
    logits = tf.layers.dense(layer5, 69, kernel_initializer=tf.contrib.layers.xavier_initializer(seed = 1))
        # Reshape output layer to 1-dim Tensor to return predictions
    predictions = tf.nn.sigmoid(logits)
    rounded_predictions = tf.round(predictions)

    # Provide an estimator spec for `ModeKeys.PREDICT`.
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"products": predictions},
            export_outputs={SIGNATURE_NAME: PredictOutput({"products": predictions})})

    # Calculate loss using custom loss function
    
    hidden_products = tf.subtract(tf.cast(labels, tf.float32),
                                  tf.cast(features[INPUT_TENSOR_NAME],
                                          tf.float32)) #now have a vector with just ones for hidden products

    hidden_mask = tf.equal(hidden_products,1)

    non_hidden_mask = tf.equal(hidden_products,0)
    
    logit_one_only = tf.boolean_mask(logits, hidden_mask)

    label_one_only = tf.boolean_mask(tf.cast(labels, tf.float32), hidden_mask)

    ones_cost = tf.reduce_sum(tf.multiply(tf.constant(69.0 ** 0.5, dtype='float32'), tf.nn.sigmoid_cross_entropy_with_logits(labels=label_one_only,
                                                       logits=logit_one_only)))


    logit_zero_only = tf.boolean_mask(logits, non_hidden_mask)

    label_zero_only = tf.boolean_mask(tf.cast(labels, tf.float32), non_hidden_mask)

    zeros_cost = tf.reduce_sum(tf.multiply(tf.constant(1.0, dtype='float32'),tf.nn.sigmoid_cross_entropy_with_logits(labels=label_zero_only,
                                                       logits=logit_zero_only)))

    loss = tf.add(ones_cost, zeros_cost)
    
    optimizer = tf.train.AdamOptimizer(
        learning_rate=params["learning_rate"])
    train_op = optimizer.minimize(
        loss=loss, global_step=tf.train.get_global_step())
    
    ###other eval metrics
    y_hat = tf.nn.sigmoid(logits)
    correct_prediction = tf.equal(tf.round(y_hat), hidden_products)

    true_positive_rate = tf.reduce_mean(tf.round(tf.boolean_mask(y_hat,hidden_mask)))

    #false_positive_rate = tf.subtract(1.0, true_positive_rate)

    false_negative_rate = tf.reduce_mean(tf.round(tf.boolean_mask(y_hat,non_hidden_mask)))

    #true_negative_rate = tf.subtract(1.0, false_negative_rate)
    #recall = tf.reduce_sum(tf.cast(correct_prediction, "float")) / tf.reduce_sum(hidden_products)

    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Calculate root mean squared error as additional eval metric
    eval_metric_ops = {
        "custom_loss" : tf.metrics.mean(loss),
        "accuracy" : tf.metrics.mean(accuracy),
        "tpr" : tf.metrics.mean(true_positive_rate),
        "fnr" : tf.metrics.mean(false_negative_rate)
    }

    # Provide an estimator spec for `ModeKeys.EVAL` and `ModeKeys.TRAIN` modes.
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


def serving_input_fn(params):
    tensor = tf.placeholder(tf.float32, shape=[1, 69])
    return build_raw_serving_input_receiver_fn({INPUT_TENSOR_NAME: tensor})()


params = {"learning_rate": LEARNING_RATE}


def train_input_fn(training_dir, params):
    xFN = os.path.join(training_dir, 'trainX')
    yFN = os.path.join(training_dir, 'trainY')
    tX = np.genfromtxt(xFN, delimiter=',')
    tY = np.genfromtxt(yFN, delimiter=',')
    return tf.estimator.inputs.numpy_input_fn(x = {INPUT_TENSOR_NAME: np.array(tX)}, y=np.array(tY), shuffle = True, num_epochs=None)()


def eval_input_fn(training_dir, params):
    xFN = os.path.join(training_dir, 'devX')
    yFN = os.path.join(training_dir, 'devY')
    tX = np.genfromtxt(xFN, delimiter=',')
    tY = np.genfromtxt(yFN, delimiter=',')
    return tf.estimator.inputs.numpy_input_fn(
        x={INPUT_TENSOR_NAME: np.array(tX)},
        y= np.array(tY),
        shuffle = True, 
        num_epochs=1)()


