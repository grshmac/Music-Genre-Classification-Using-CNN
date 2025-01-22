import tensorflow as tf
import numpy as np

#building our cnn model
class CNNModel:
    def __init__(self, input_shape, num_classes):
        stddev = 0.01  #this is standard deviation for weight initialization

        self.input_shape = input_shape
        self.num_classes = num_classes

        self.weights = {
            'conv1': tf.Variable(
                tf.random.normal(
                    [3, 3, input_shape[-1], 32], stddev=stddev
                    )
                ),
            'conv2': tf.Variable(
                tf.random.normal(
                    [3, 3, 32, 32], stddev=stddev
                    )
                ),
            'conv3': tf.Variable(
                tf.random.normal(
                    [3, 3, 32, 64], stddev=stddev
                    )
                ),
            'conv4': tf.Variable(
                tf.random.normal(
                    [3, 3, 64, 64], stddev=stddev
                    )
                ),
            'conv5': tf.Variable(
                tf.random.normal(
                    [3, 3, 64, 128], stddev=stddev
                    )
                ),
            'conv6': tf.Variable(
                tf.random.normal(
                    [3, 3, 128, 128], stddev=stddev
                    )
                ),

            #since fully connected layer weights will be initialized later based on computed size
            'fc1': None,
            'fc2': tf.Variable(
                tf.random.normal(
                    [1200, num_classes], stddev=stddev))
        }

        self.biases = {
            'conv1': tf.Variable(tf.zeros([32])),
            'conv2': tf.Variable(tf.zeros([32])),
            'conv3': tf.Variable(tf.zeros([64])),
            'conv4': tf.Variable(tf.zeros([64])),
            'conv5': tf.Variable(tf.zeros([128])),
            'conv6': tf.Variable(tf.zeros([128])),
            'fc1': None,
            'fc2': tf.Variable(tf.zeros([num_classes]))
        }

    def build_fc1(self, x):
        #to dynamically calculate the flattened size
        flatten_dim = np.prod(x.shape[1:])  #total size after flattening
        self.weights['fc1'] = tf.Variable(tf.random.normal([flatten_dim, 1200], stddev=0.01))
        self.biases['fc1'] = tf.Variable(tf.zeros([1200]))

    def forward(self, x, is_training=True):
        #convolutional layers with ReLU, Max Pooling, and Dropout
        x = tf.nn.conv2d(x, self.weights['conv1'], strides=1, padding='SAME') + self.biases['conv1']
        x = tf.nn.relu(x)
        x = tf.nn.conv2d(x, self.weights['conv2'], strides=1, padding='VALID') + self.biases['conv2']
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')

        x = tf.nn.conv2d(x, self.weights['conv3'], strides=1, padding='SAME') + self.biases['conv3']
        x = tf.nn.relu(x)
        x = tf.nn.conv2d(x, self.weights['conv4'], strides=1, padding='VALID') + self.biases['conv4']
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')

        x = tf.nn.conv2d(x, self.weights['conv5'], strides=1, padding='SAME') + self.biases['conv5']
        x = tf.nn.relu(x)
        x = tf.nn.conv2d(x, self.weights['conv6'], strides=1, padding='VALID') + self.biases['conv6']
        x = tf.nn.relu(x)
        x = tf.nn.max_pool2d(x, ksize=2, strides=2, padding='SAME')

        if is_training:
            x = tf.nn.dropout(x, rate=0.3)

        #flatten
        x = tf.reshape(x, [x.shape[0], -1])

        #initialize fully connected layer weights if not already done
        if self.weights['fc1'] is None:
            self.build_fc1(x)

        #fully connected layers
        x = tf.matmul(x, self.weights['fc1']) + self.biases['fc1']
        x = tf.nn.relu(x)

        if is_training:
            x = tf.nn.dropout(x, rate=0.4)

        x = tf.matmul(x, self.weights['fc2']) + self.biases['fc2']
        return x  #return raw logits
