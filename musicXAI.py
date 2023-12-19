import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sys
import os
import shap
import pickle
import librosa
import librosa.display
from IPython.display import Audio
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from lime import lime_tabular
import time

df = pd.read_csv("C:\\Users\\antho\\Desktop\\musicXAI\\Data\\features_3_sec.csv")



df = df.drop(labels='filename', axis=1)

audio_recording = "C:\\Users\\antho\\Desktop\\musicXAI\\Data\\genres_original\\blues\\blues.00003.wav"
data, sr = librosa.load(audio_recording)

plt.figure(figsize=(12, 4))
librosa.display.waveshow(data, sr=sr, color="#8B008B")
plt.title('Amplitude of the Blues00003 example Wave File')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

stft = librosa.stft(data)
stft_magnitude = np.abs(stft)
stft_db = librosa.amplitude_to_db(stft_magnitude)

plt.figure(figsize=(14, 6))
librosa.display.specshow(stft_magnitude, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
plt.title('Spectrogram of the Music File (magnitude)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.show()

plt.figure(figsize=(14, 6))
librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar()
plt.title('Spectrogram of the Music File (db)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.show()

class_list = df.iloc[:,-1]
convertor = LabelEncoder()
y = convertor.fit_transform(class_list)

fit = StandardScaler()
x = fit.fit_transform(np.array(df.iloc[:,:-1]))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

print("x_train length: ", len(x_train))
print("y_train length: ", len(y_train))
print("x_test length: ", len(x_test))
print("y_test length: ", len(y_test))

def train(model, epochs, optimizer):
    batch_size = 128
    model.compile(optimizer=optimizer, loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')

    start_time = time.time()
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs = epochs, batch_size = batch_size)
    end_time = time.time()
    print("Training time: ", end_time - start_time, "seconds")

    return model, history

def validate(history):
    print("Val acc: ", max(history.history["val_accuracy"]))
    pd.DataFrame(history.history).plot(figsize=(12,6))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

def shap_explainability(model, background_data, instance):
    explainer = shap.Explainer(model, background_data)
    shap_values = explainer.shap_values(instance)
    shap.summary_plot(shap_values, instance, feature_names=df.columns[:-1])

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

print("x train shape: ", x_train.shape[1],)
model, history = train(model, epochs=100, optimizer='adam')

validate(history)

background_data = x_train[:100]
instance_to_explain = x_test[0]
instance_to_explain = np.reshape(instance_to_explain, (1, -1))
shap_explainability(model, background_data, instance_to_explain)

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(x_test, y_test, batch_size=128)
print("Loss is: ", test_loss)
print("Test acc is: ", test_acc * 100, "%")

explainer = lime_tabular.LimeTabularExplainer(training_data=np.array(x_train), 
                                              feature_names=list(df.columns[:-1]), 
                                              class_names=convertor.classes_, 
                                              discretize_continuous=True)

#####################################################################################

instance = x_test[0] # Explain some test data

probs = model.predict(instance.reshape(1, -1))
top_class = np.argmax(probs, axis=1)

exp = explainer.explain_instance(instance, lambda x: keras.activations.softmax(tf.convert_to_tensor(model.predict(x))).numpy(), top_labels=1)

print(f"\nClass Explanation {convertor.classes_[top_class[0]]}:")
exp.save_to_file(f'ClassExplanation_{convertor.classes_[top_class[0]]}.html')

#####################################################################################

instance = x_test[1]

probs = model.predict(instance.reshape(1, -1))
top_class = np.argmax(probs, axis=1)

exp = explainer.explain_instance(instance, lambda x: keras.activations.softmax(tf.convert_to_tensor(model.predict(x))).numpy(), top_labels=1)

print(f"\nExplanation for class {convertor.classes_[top_class[0]]}:")
exp.save_to_file(file_path=f'62Explanation_{convertor.classes_[top_class[0]]}.html')

#####################################################################################

explainer = lime_tabular.LimeTabularExplainer(training_data=np.array(x_train), 
                                              feature_names=list(df.columns[:-1]), 
                                              class_names=convertor.classes_, 
                                              discretize_continuous=True)

instance = x_test[2]


probs = model.predict(instance.reshape(1, -1))
top_class = np.argmax(probs, axis=1)
exp = explainer.explain_instance(instance, model.predict, top_labels=1)

print(f"\nClass Explanation {convertor.classes_[top_class[0]]}:")
exp.save_to_file(file_path=f'Class Explanation_{convertor.classes_[top_class[0]]}.html')

#####################################################################################
explainer = lime_tabular.LimeTabularExplainer(training_data=np.array(x_train), 
                                              feature_names=list(df.columns[:-1]), 
                                              class_names=convertor.classes_, 
                                              discretize_continuous=True)

instance = x_test[3]

probs = model.predict(instance.reshape(1, -1))

top_class = np.argmax(probs, axis=1)

exp = explainer.explain_instance(instance, model.predict, top_labels=1)
print(f"\nClass Explanation {convertor.classes_[top_class[0]]}:")
exp.save_to_file(file_path=f'Class Explanation_{convertor.classes_[top_class[0]]}.html')






