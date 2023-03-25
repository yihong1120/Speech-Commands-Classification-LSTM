# Speech-Commands-Classification-LSTM

This is an open-source project that demonstrates how to build and train a deep learning model to recognize speech commands using TensorFlow. The model is trained on the [Speech Commands dataset](https://www.tensorflow.org/datasets/catalog/speech_commands), which consists of over 105,000 WAV audio files of people saying different commands such as "yes", "no", "stop", "go", etc.

## Getting Started

To run the script in this repository, you will need a Unix based system with Bash and Python installed. You should also have a Python script that you want to automate the execution of.

## Requirements

To run this project, you need to have the following software installed:

* Python 3.6 or higher
* TensorFlow 2.0 or higher
* NumPy
* librosa
* TensorFlow Datasets

You can install these packages by running the following command:

    pip install tensorflow numpy librosa tensorflow-datasets


## Usage

### Training

You can train the model by running the speech_commands_classification.py script. This script will download the Speech Commands dataset, preprocess the data, build the model, and train the model on the dataset. To run the script, open a terminal and navigate to the project directory, then run:

    python speech_commands_classification.py

This will start the training process, which may take several hours depending on your hardware. You can adjust the batch_size and epochs parameters in the script to control the training process.

### Prediction

You can make predictions on new audio files by running the speech_command_prediction.py script. This script loads a pre-trained model from a saved file and uses it to predict the label of a new audio file. To run the script, open a terminal and navigate to the project directory, then run:

    python speech_command_prediction.py


This will load the pre-trained model and make a prediction on a sample audio file. You can modify the script to use your own audio file by changing the filename variable.

### Google Colab

If you have access to a GPU, you can use Google Colab to train and run the model. We have provided a Jupyter Notebook file (speech_commands_classification.ipynb) that you can upload to Google Colab and run. To use Google Colab, follow these steps:

1. Upload the speech_commands_classification.ipynb file to your Google Drive.
2. Open the file in Google Colab.
3. Change the runtime type to "GPU" by going to "Runtime" > "Change runtime type" and selecting "GPU" from the "Hardware accelerator" dropdown.
4. Run the cells in the notebook to train and evaluate the model.

## Contributors

* [yihong1120](https://github.com/yihong1120) - Project Lead

## Conclusion

This script provides an easy way to automate a specific set of actions on the screen at specific times on weekdays only. The use of pyautogui and shell scripting makes it easy to automate these actions without needing to manually perform them each time.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/yihong1120/Speech-Commands-Classification-LSTM/blob/main/LICENSE) file for details.
