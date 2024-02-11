import shap
import librosa
import numpy as np
from keras.models import load_model
from keras.models import model_from_json

# Load the model architecture from JSON file
json_file = open('../model/CNN_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# Create a new Sequential model and load the architecture
model = model_from_json(loaded_model_json)

# Load the weights into the new model
model.load_weights('../model/CNN_model_weights.h5')

# Define a function to preprocess audio data for the model
def preprocess_audio(file_path, duration=2.5, offset=0.6):
    data, sr = librosa.load(file_path, duration=duration, offset=offset)
    # Additional processing steps may be required based on your model's input requirements
    return data, sr

# Load an example audio file
example_audio_path = '../download.wav'  # Replace with your audio file path
audio_data, sr = preprocess_audio(example_audio_path)

# Modify the desired shape for multiple instances
desired_shape_multiple = (5, 2376)  # Change num_instances to the desired number
audio_data_reshaped_multiple = np.resize(audio_data, desired_shape_multiple)

# Create the explainer using the KernelExplainer
explainer_multiple = shap.KernelExplainer(model.predict, data=audio_data_reshaped_multiple)

# Get SHAP values for multiple instances
shap_values_multiple = explainer_multiple.shap_values(audio_data_reshaped_multiple)

# Plot SHAP summary plot for multiple instances
shap.summary_plot(shap_values_multiple, audio_data_reshaped_multiple, plot_type='bar')
