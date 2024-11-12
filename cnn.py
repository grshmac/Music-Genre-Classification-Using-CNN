import numpy as np
from cnn_layers.conv_layer import ConvLayer
from cnn_layers.pooling_layer import PoolingLayer
from cnn_layers.fully_connected_layer import FullyConnectedLayer
from cnn_layers.softmax_layer import SoftmaxLayer

# Load preprocessed data
train_data = np.load("train_data.npz")
val_data = np.load("val_data.npz")
test_data = np.load("test_data.npz")

X_train, y_train = train_data["data"], train_data["labels"]
X_val, y_val = val_data["data"], val_data["labels"]
X_test, y_test = test_data["data"], test_data["labels"]

# Initialize and train CNN 



# Example placeholder for training loop
for epoch in range(num_epochs):
    for batch in generate_batches(X_train, y_train, batch_size=32):
        # Forward and backward passes for each batch
        pass

# After training, evaluate on validation and test sets
