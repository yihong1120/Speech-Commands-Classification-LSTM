import numpy as np
import tensorflow as tf
import librosa

# Load the audio file to be predicted
filename = "path/to/audio/file.wav"
audio, sr = librosa.load(filename, sr=16000)

# Load the audio file to be predicted
model = tf.keras.models.load_model("path/to/audio/model/file.h5")

# Create the model input
audio = tf.expand_dims(audio, axis=0)
audio = tf.expand_dims(audio, axis=-1)

# Make predictions
prediction = model.predict(audio)
predicted_label = np.argmax(prediction)