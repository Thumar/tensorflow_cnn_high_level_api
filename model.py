import math

import tensorflow as tf


# Model
def model_fn(features, labels, mode):
    def learn_rate(lr, step):
        return 0.0001 + tf.train.exponential_decay(lr, step, 800, 1 / math.e)

    input_layer = tf.reshape(features["image"], [-1, 20, 20, 3])
    input_layer = tf.to_float(input_layer) / 255.0

    Y_ = labels

    # 1 layer [filter_size:4x4,stride:1,padding:0,filters:16]
    conv1 = tf.layers.conv2d(input_layer, filters=16, kernel_size=[4, 4], strides=1, padding="same", activation=None,
                             use_bias=False)
    batch_norm1 = tf.layers.batch_normalization(conv1, axis=-1, momentum=0.993, epsilon=1e-5, center=True,
                                                scale=False, training=(mode == tf.estimator.ModeKeys.TRAIN))
    relu1 = tf.nn.relu(batch_norm1)

    # 2 layer
    conv2 = tf.layers.conv2d(relu1, filters=32, kernel_size=[3, 3], strides=2, padding="same", activation=None,
                             use_bias=False)
    batch_norm2 = tf.layers.batch_normalization(conv2, axis=-1, momentum=0.993, epsilon=1e-5, center=True,
                                                scale=False, training=(mode == tf.estimator.ModeKeys.TRAIN))
    relu2 = tf.nn.relu(batch_norm2)

    # 3 layer
    conv3 = tf.layers.conv2d(relu2, filters=64, kernel_size=[2, 2], strides=2, padding="same", activation=None,
                             use_bias=False)
    batch_norm3 = tf.layers.batch_normalization(conv3, axis=-1, momentum=0.993, epsilon=1e-5, center=True,
                                                scale=False, training=(mode == tf.estimator.ModeKeys.TRAIN))
    relu3 = tf.nn.relu(batch_norm3)

    # Flatten all values for fully connected layer
    relu3_flat = tf.reshape(relu3, [-1, 4 * 16 * 5 * 5])

    # Dense Layer
    dense = tf.layers.dense(relu3_flat, 80, activation=None, use_bias=False)
    batch_norm_dense = tf.layers.batch_normalization(dense, axis=-1, momentum=0.993, epsilon=1e-5, center=True,
                                                     scale=False, training=(mode == tf.estimator.ModeKeys.TRAIN))
    relu_dense = tf.nn.relu(batch_norm_dense)

    # Logits Layer
    Ylogits = tf.layers.dense(relu_dense, 2)

    predictions = {
        "classes": tf.argmax(input=Ylogits, axis=1),
        "probabilities": tf.nn.softmax(Ylogits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    predict = tf.nn.softmax(Ylogits)
    classes = tf.cast(tf.argmax(predict, 1), tf.uint8)

    # Calculate Loss for TRAIN and EVAL modes
    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(tf.one_hot(Y_, 2), Ylogits)) * 100

    # Configure the Training Op for TRAIN mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), learning_rate=0.01,
                                                   optimizer="Adam", learning_rate_decay_fn=learn_rate)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics
    eval_metrics = {'accuracy': tf.metrics.accuracy(classes, Y_)}

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metrics)
