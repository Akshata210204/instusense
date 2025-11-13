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


# Quick check if running this file directly
if __name__ == "__main__":
    dummy_model = build_bilstm((1, 100), 3)  # Example with 100 features
    dummy_model.summary()
