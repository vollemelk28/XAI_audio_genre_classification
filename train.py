import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
import pickle

df = pd.read_csv("C:\\Users\\antho\\Desktop\\musicXAI\\Data\\features_3_sec.csv")
df = df.drop(labels='filename', axis=1)

class_list = df.iloc[:,-1]
convertor = LabelEncoder()
y = convertor.fit_transform(class_list)

fit = StandardScaler()
x = fit.fit_transform(np.array(df.iloc[:,:-1]))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

def train(model, epochs, optimizer):
    batch_size = 128
    model.compile(optimizer=optimizer, loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')

    start_time = time.time()
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs, batch_size = batch_size)
    end_time = time.time()
    print("Training time: ", end_time - start_time, "seconds")

    return model, history, x_train, x_test, y_train, y_test, convertor

model = keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(x_train.shape[1],)),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(10, activation='softmax'),
])

model, history, x_train, x_test, y_train, y_test, convertor = train(model, epochs=100, optimizer='adam')

model.save('trained_model.h5')


with open('train_data.pkl', 'wb') as f:
    pickle.dump([history, x_train, x_test, y_train, y_test, convertor], f)
