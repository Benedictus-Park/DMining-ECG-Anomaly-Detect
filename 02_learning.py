import os
import wfdb
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

tmp_y = pd.read_csv('dataset/label.csv')

X = np.array([wfdb.rdsamp('dataset/' + f)[0] for f in tmp_y.fname])
Y = []

for i in tmp_y[['class']].values:
    # NORM, MI(STEMI), STTC(NSTEMI), CD(부정맥), HYP(비대), ETC
    if i[0] == 0: # 정상(NORM)
        Y.append(0)
    else: # 비정상(ETCs)
        Y.append(1)

Y = np.array(Y)
X = tf.cast(X, tf.float32)

### Algorithm 1 ###
model_pretty = tf.keras.Sequential([
    # 1D Convolutional Layer
    layers.Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),

    # LSTM Layers
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Bidirectional(layers.LSTM(32, return_sequences=False)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    # Dense Layers
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.25),
    
    layers.Dense(1, activation='sigmoid')
])

model_pretty.summary()
model_pretty.compile(
    loss='binary_crossentropy', 
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    metrics=['accuracy'])

record_pretty_batch128 = model_pretty.fit(X, Y,
                   epochs=28,
                   batch_size=128,
                   validation_split=0.2,
                   shuffle=True)

model_pretty.save("pretty-batch128.keras")

fig, ax = plt.subplots()
plt.plot(record_pretty_batch128.history['loss'], label='Training Loss')
plt.plot(record_pretty_batch128.history['val_loss'], label='Validation Loss')
plt.legend()
plt.suptitle('Loss')
plt.show()

plt.plot(record_pretty_batch128.history['accuracy'], label='Training Accuracy')
plt.plot(record_pretty_batch128.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.suptitle('Accuracy')
plt.show()

### Algorithm 2 ###
model = tf.keras.Sequential([
    # 1D Convolutional Layer
    layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], X.shape[2])),
    layers.BatchNormalization(),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.3),

    # LSTM Layers
    layers.LSTM(64, return_sequences=True),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Bidirectional(layers.LSTM(32, return_sequences=False)),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    # Dense Layers
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.1),
    
    layers.Dense(1, activation='sigmoid')
])

model.summary()
model.compile(
    loss='binary_crossentropy', 
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
    metrics=['accuracy'])

record = model.fit(X, Y,
                   epochs=26,
                   batch_size=128,
                   validation_split=0.2,
                   shuffle=True)

model.save("algorithm2.keras")

fig, ax = plt.subplots()
plt.plot(record.history['loss'], label='Training Loss')
plt.plot(record.history['val_loss'], label='Validation Loss')
plt.legend()
plt.suptitle('Loss')
plt.show()

plt.plot(record.history['accuracy'], label='Training Accuracy')
plt.plot(record.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.suptitle('Accuracy')
plt.show()