import os
import sys
from os import listdir
from os.path import join, isfile

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging

import model

logging.set_verbosity(logging.INFO)
logging.log(logging.INFO, "Tensorflow version " + tf.__version__)

DATA_DIR = "/home/manu/PycharmProjects/DataSet/planesnet/planesnet"

image_files = [join(DATA_DIR, f) for f in listdir(DATA_DIR) if isfile(join(DATA_DIR, f))]

TEST_SIZE = 5000

test_file = image_files[:TEST_SIZE]
train_file = image_files[TEST_SIZE:]


def load_dataset(filename):
    labels = list(map(lambda filename: int(os.path.basename(filename)[0:1] == '1'), filename))
    return tf.data.Dataset.from_tensor_slices((tf.constant(filename), tf.constant(labels)))


def load(filename, label):
    return tf.read_file(filename), label


def decode(img_bytes, label):
    img_decoded = tf.image.decode_image(img_bytes, channels=3)
    return img_decoded, label


def features_and_labels(dataset):
    it = dataset.make_one_shot_iterator()
    images, labels = it.get_next()
    features = {'image': images}
    return features, labels


def dataset_input_fn(dataset):
    dataset = dataset.map(load)
    dataset = dataset.map(decode)
    dataset = dataset.shuffle(20)
    dataset = dataset.batch(1)
    dataset = dataset.repeat()
    return features_and_labels(dataset)


def dataset_eval_input_fn(dataset):
    dataset = dataset.map(load)
    dataset = dataset.map(decode)
    dataset = dataset.batch(TEST_SIZE)
    return features_and_labels(dataset)


def main(argv):
    training_config = tf.contrib.learn.RunConfig(save_checkpoints_secs=None, save_checkpoints_steps=500)

    dataset_train = load_dataset(train_file)
    dataset_test = load_dataset(test_file)

    estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir="/tmp/cnn_data", config=training_config)

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    estimator.train(input_fn=lambda: dataset_input_fn(dataset_train),
                    steps=10000,
                    hooks=[logging_hook])

    eval_results = estimator.evaluate(input_fn=lambda: dataset_eval_input_fn(dataset_test))
    print(eval_results)


if __name__ == '__main__':
    main(sys.argv)
