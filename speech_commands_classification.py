import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

# Load the Speech Commands dataset
train_dataset, validation_dataset, test_dataset = tfds.load(
    "speech_commands",
    split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
    shuffle_files=True,
    as_supervised=True
)
# Define parameters
batch_size = 32
epochs = 10 # Shorten the training time for demonstration purposes
num_classes = 35 # There are 35 different commands in the Speech Commands dataset
frame_length = 16000 # Set the frame_length to 16000 because the audio sampling rate is 16 kHz

# Define the data preprocessing function
def preprocess_dataset(audio, label):
    # Cast the data type using tf.cast
    audio = tf.cast(audio, tf.float32)
    # If the length of the audio file is less than the fixed frame length, pad it with zeros
    if tf.shape(audio)[0] < frame_length:
        padding = tf.zeros((frame_length - tf.shape(audio)[0],), dtype=tf.float32)
        audio = tf.concat([audio, padding], axis=0)
    # If the length of the audio file is greater than the fixed frame length, trim it to the fixed frame length
    elif tf.shape(audio)[0] > frame_length:
        audio = audio[:frame_length]
    else:
        audio = audio
    # Use tf.numpy_function to call a NumPy function to process the tensor
    audio = tf.numpy_function(func=np.int16, inp=[audio], Tout=tf.int16)
    # Trim the audio file to the fixed frame length; frame_length has been set to 16000 here
    frame_step = frame_length // 2
    audio = tf.signal.frame(audio, frame_length, frame_step, pad_end=True)
    # Convert the label to a one-hot vector
    label = tf.one_hot(label, depth=num_classes)
    return audio, label

# Map and preprocess the dataset
train_data = train_dataset.map(preprocess_dataset)
train_data = train_data.shuffle(buffer_size=1024).batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)

validation_data = validation_dataset.map(preprocess_dataset)
validation_data = validation_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_data = test_dataset.map(preprocess_dataset)
test_data = test_data.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Define the model, optimizer, and loss function
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(None, frame_length)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss = tf.keras.losses.CategoricalCrossentropy()

# Compile the model
model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

# Train the model
train_size = tf.data.experimental.cardinality(train_dataset).numpy()
steps_per_epoch = int(train_size / batch_size)
model.fit(train_data, epochs=epochs, validation_data=validation_data, steps_per_epoch=steps_per_epoch)

# Evaluate the model
model.evaluate(test_data)
