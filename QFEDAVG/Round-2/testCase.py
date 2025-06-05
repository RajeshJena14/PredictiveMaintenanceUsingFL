import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from sklearn.metrics import f1_score

# Define model architecture from client.py
def build_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,), kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(6, activation='softmax')
    ])
    return model

# Load test data and weights
try:
    X_test = np.load("C:/Users/rkjra/Desktop/FL/QFEDAVG/X_test.npy")
    y_test = np.load("C:/Users/rkjra/Desktop/FL/QFEDAVG/y_test.npy")
    weights = np.load("C:/Users/rkjra/Desktop/FL/QFEDAVG/final_weights.npy", allow_pickle=True)
except FileNotFoundError as e:
    print(f"Error: Could not load file - {e}")
    exit(1)

# Build model
model = build_model(input_dim=6)

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Set weights
try:
    model.set_weights(weights)
except ValueError as e:
    print(f"Error: Could not set model weights - {e}")
    exit(1)

# Evaluate model
try:
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    f1 = f1_score(y_test, y_pred_classes, average='macro')
    print(f"Test Loss: {loss:.4f}, Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
except Exception as e:
    print(f"Error during evaluation: {e}")
    exit(1)
