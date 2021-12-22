# Imports
import random
import librosa
import pandas as pd
import os
from keras.models import load_model
import numpy as np
from tqdm import tqdm
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def predict_audio(audio):
    audio = audio.reshape(1, -1)
    index = np.argmax(model.predict(audio))
    return classes[index]


def model_layer(_model, units, act, drop):
    _model.add(Dense(units, input_shape=(40,)))
    _model.add(Activation(act))
    _model.add(Dropout(drop))


def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
    mfcc_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfcc_scaled_features = np.mean(mfcc_features.T, axis=0)

    return mfcc_scaled_features


### Now we iterate through every audio file and extract features
### using Mel-Frequency Cepstral Coefficients
dataset_path = "./Dataset"
labels = os.listdir(dataset_path)
extracted_features = []
for label in tqdm(labels):
    file_path = dataset_path + '/' + label + '/wav/'
    for file_name in os.listdir(file_path):
        features = features_extractor(file_path + file_name)
        # Append each record features with its Label/Class
        extracted_features.append([features, label])

### converting extracted_features to Pandas dataframe
extracted_features_df = pd.DataFrame(extracted_features, columns=['feature', 'class'])
extracted_features_df.head()

### Split the dataset into independent and dependent dataset
x = np.array(extracted_features_df['feature'].tolist())
y = np.array(extracted_features_df['class'].tolist())

# Classes encoding
le = LabelEncoder()
y = le.fit_transform(y)
classes = list(le.classes_)

# split the data
x_tr, x_val, y_tr, y_val = train_test_split(x, y, stratify=y, test_size=0.2, random_state=777, shuffle=True)

model = Sequential()
# first layer
model_layer(model, 100, 'relu', 0.5)
# second layer
model_layer(model, 200, 'relu', 0.5)
# third layer
model_layer(model, 100, 'relu', 0.5)
# final layer
model_layer(model, 40, 'softmax', 0)

model.summary()
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Train the model
model_path = './saved_models/audio_classification.hdf5'
num_epochs = 200
num_batch_size = 32
checkpointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)
model.fit(x_tr, y_tr, batch_size=num_batch_size, epochs=num_epochs, validation_data=(x_val, y_val), callbacks=[checkpointer], verbose=1)

# find the accuracy
test_accuracy = model.evaluate(x_val, y_val, verbose=0)
print(test_accuracy[1])

# Load model
model = load_model(model_path)

# Start predict
predictions_num = 3
for i in range(predictions_num):
    index = random.randint(0, len(x_val) - 1)
    print("Audio: ", classes[y_val[index]])
    print("Text: ", predict_audio(x_val[index]))