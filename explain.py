import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import load_model
from lime import lime_tabular
import shap
import pickle

df = pd.read_csv("C:\\Users\\antho\\Desktop\\musicXAI\\Data\\features_3_sec.csv")

def shap_explainability(model, background_data, instance):
    explainer = shap.Explainer(model, background_data)
    shap_values = explainer.shap_values(instance)
    shap.summary_plot(shap_values, instance, feature_names=df.columns[:-1])

model = load_model('trained_model.h5')

with open('train_data.pkl', 'rb') as f:
    history, x_train, x_test, y_train, y_test, convertor = pickle.load(f)

background_data = x_train[:100]
instance_to_explain = x_test[0]
instance_to_explain = np.reshape(instance_to_explain, (1, -1))
shap_explainability(model, background_data, instance_to_explain)

explainer = lime_tabular.LimeTabularExplainer(training_data=np.array(x_train), 
                                              feature_names=list(df.columns[:-1]), 
                                              class_names=convertor.classes_, 
                                              discretize_continuous=True)

test_selection = x_test[:4]

for i, test in enumerate(test_selection):
    probs = model.predict(test.reshape(1, -1))
    top_class = np.argmax(probs, axis=1)
    exp = explainer.explain_instance(test, lambda x: keras.activations.softmax(tf.convert_to_tensor(model.predict(x))).numpy(), top_labels=1)
    print(f"\nClass Explanation {convertor.classes_[top_class[0]]}:")
    exp.save_to_file(f'ClassExplanation_{convertor.classes_[top_class[0]]}_{i}.html')
