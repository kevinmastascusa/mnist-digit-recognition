# Importing the necessary TensorFlow modules
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data by scaling it to the range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# Define the architecture of the neural network
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Reshape the 28x28 images into a 1D array
    Dense(128, activation='relu'),  # Add a fully connected layer with 128 neurons and ReLU activation
    Dense(10, activation='softmax')  # Add the output layer with 10 neurons and softmax activation
])

# Compile the model with the Adam optimizer, sparse categorical cross-entropy loss, and accuracy metric
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train the model on the training data for 5 epochs with a batch size of 32
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluate the trained model on the test data and obtain the test accuracy
_, test_accuracy = model.evaluate(x_test, y_test)

# Print the test accuracy to the console
print('Test accuracy:', test_accuracy)
