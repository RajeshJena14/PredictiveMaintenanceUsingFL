from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.regularizers import l2

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
weights = np.load("final_weights.npy", allow_pickle=True)

model = Sequential([
    Dense(256, activation='relu', input_shape=(6,), kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(6, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.set_weights(weights)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
f1 = f1_score(y_test, y_pred_classes, average='macro')
print(f"Test Loss: {loss:.4f}, Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
