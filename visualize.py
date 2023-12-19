import matplotlib.pyplot as plt
import pandas as pd
import librosa
import librosa.display
import pickle

df = pd.read_csv("C:\\Users\\antho\\Desktop\\musicXAI\\Data\\features_3_sec.csv")

music_fragment = "C:\\Users\\antho\\Desktop\\musicXAI\\Data\\genres_original\\blues\\blues.00003.wav"
data, sr = librosa.load(music_fragment, mono=True)

plt.figure(figsize=(14, 6))
librosa.display.waveshow(data, sr=sr, color="#8B008B")
plt.title('Amplitude of the Blues00003 example Wave File')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

def validate(history):
    print("Val acc: ", max(history.history["val_accuracy"]))
    pd.DataFrame(history.history).plot(figsize=(14,6))
    plt.xlabel('# Epochs')
    plt.ylabel('Accuracy')
    plt.show()

with open('train_data.pkl', 'rb') as f:
    history, x_train, x_test, y_train, y_test, convertor = pickle.load(f)

validate(history)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()
