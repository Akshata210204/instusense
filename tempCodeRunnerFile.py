import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Bidirectional, LSTM

def build_bilstm(input_shape, num_classes):
    """
    Build and return a BiLSTM model for intrusion detection.

    Args:
        input_shape (tuple): Shape of the input data (timesteps, features).
                             For example: (1, num_features)
        num_classes (int): Number of output classes (e.g., 3 for Normal/DoS/Probe).

    Returns:
        model (tf.keras.Model): Compiled BiLSTM model.
    """

    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=False), input_shape=input_shape),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])

    return model