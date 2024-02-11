import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model, model_from_json
import pickle
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import streamlit as st
import sounddevice as sd
import wavio
import time
import speech_recognition as sr

import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = st.secrets["google"]["api_key"]

# Set parameters for recording
duration = 5  # seconds
fs = 44100  # sample rate
channels = 2  # number of audio channels (stereo)

def record_audio():
    st.text("Recording audio...")
    audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=channels, blocking=True)
    filename = "./recorded/recorded_audio1.wav"
    sf.write(filename, audio_data, fs)
    st.text(f"Audio recorded and saved to {filename}")
    return filename


# Load the CNN model architecture from JSON file
json_file = open('./model/CNN_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# Create a new Sequential model and load the architecture
loaded_model = tf.keras.models.Sequential()
loaded_model = model_from_json(loaded_model_json)

# Load the weights into the new model
loaded_model.load_weights("./model/best_model1_weights.h5")
print("Loaded model from disk")

# Load scaler and encoder
with open('./model/scaler2.pickle', 'rb') as f:
    scaler2 = pickle.load(f)

with open('./model/encoder2.pickle', 'rb') as f:
    encoder2 = pickle.load(f)

print("Done")

def zcr(data, frame_length, hop_length):
    zcr_values = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr_values)

def rmse(data, frame_length=2048, hop_length=512):
    rmse_values = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse_values)

def mfcc(data, sr, frame_length=2048, hop_length=512, flatten=True):
    mfcc_values = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc_values.T) if not flatten else np.ravel(mfcc_values.T)

def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.array([])
    result = np.hstack((result, zcr(data, frame_length, hop_length), rmse(data, frame_length=frame_length, hop_length=hop_length), mfcc(data, sr, frame_length, hop_length)))
    # print("result", result)
    return result

def get_predict_feat(path):
    # Load audio data using librosa
    d, s_rate = librosa.load(path, duration=2.5, offset=0.6)
    
    # Extract features
    res = extract_features(d)
    
    # Reshape and transform features
    result = np.reshape(res, newshape=(1, -1))
    i_result = scaler2.transform(result)
    final_result = np.expand_dims(i_result, axis=2)
    return final_result

emotion_weights = {
    "angry": 0.725,
    "disgust": 0.5775,
    "fear": 0.945,
    "happy": 0.0825,
    "neutral": 0.2475,
    "sad": 0.835,
    "surprise": 0.4125
}

def prediction(path1):
    res = get_predict_feat(path1)
    
    # Predict using the loaded model
    predictions = loaded_model.predict(res)
    
    # Inverse transform predictions using the encoder
    decoded_predictions = encoder2.inverse_transform(predictions)

    # Get the feature names from the OneHotEncoder
    feature_names = encoder2.get_feature_names_out()
    
    # Extract emotion names from feature names
    emotion_names = [feature.split('_')[-1] for feature in feature_names]

    # Create a dictionary mapping emotion names to their scores
    scores = {emotion: score for emotion, score in zip(emotion_names, predictions.flatten())}

    # Print all emotions with their scores
    for emotion, score in scores.items():
        print(f"{emotion}: {score}")

    # Calculate the weighted sum
    weighted_sum = sum(score * emotion_weights[emotion] for emotion, score in scores.items())*100
    print("Weighted Sum:", weighted_sum, "Stress Level")

    # Return the predicted emotion
    return decoded_predictions[0][0], weighted_sum


import joblib

ensemble_model = joblib.load('./model/ensemble2.pkl')
scaler = joblib.load('./model/scaler1.pkl')

import librosa
import numpy as np

def extract_audio_features(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file)

    # Extract features
    # Speech rate (number of speech segments divided by duration)
    speech_rate = len(librosa.effects.split(y)) / librosa.get_duration(y=y, sr=sr)

    # Energy
    energy = np.mean(librosa.feature.rms(y=y))

    # Zero crossing rate
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))

    # Pitch-related features
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_range = np.max(pitches) - np.min(pitches)

    # Spectral features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    # Combine features into a single feature vector
    features = [
        speech_rate,
        energy,
        zero_crossing_rate,
        pitch_range,
        spectral_centroid,
        spectral_bandwidth,
        spectral_rolloff
    ]

    return features


import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
import os
import getpass
from dotenv import load_dotenv
load_dotenv()
os.getenv('GOOGLE_API_KEY')

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Provide Api key")


import google.generativeai as palm


import pyttsx3

def text_to_speech(text):
    engine = pyttsx3.init()

    engine.setProperty('rate', 150)  
    engine.setProperty('volume', 0.9)  

    engine.say(text)

    engine.runAndWait()


class SessionState:
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)



# Streamlit app
def main():
    st.title("Medical Emergency Assistance App")
    session_state = SessionState(audio_history=[])

    # Record audio button
    if st.button("Record Audio"):
        # filename = record_audio()

        # Predict emotion and stress level
        emotion, stress = prediction('./audios/test2.mp3')

        features = extract_audio_features('./audios/test2.mp3')

        # Standardize features using the same scaler used during training
        X_unknown = np.array(features).reshape(1, -1)  # Reshape to a 2D array as expected by the scaler
        X_unknown_scaled = scaler.transform(X_unknown)

        # Use the trained stacking ensemble model to predict urgency level
        urgency_level = ensemble_model.predict(X_unknown_scaled)[0]
        st.text(f"Predicted Urgency Level: {urgency_level}")
        
        # Display results
        st.text(f"Predicted Emotion: {emotion}")
        st.text(f"Stress Level: {stress}")
        HUGGING_FACE_API_KEY = st.secrets["huggingface"]["api_key"]
        API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
        headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}

        def query(filename):
            with open(filename, "rb") as f:
                data = f.read()
            response = requests.post(API_URL, headers=headers, data=data)
            return response.json()
        # Query text API
        text = query('./audios/test2.mp3')
        print(text)
        st.text(f"Text from API: {text}")

        # Generate emergency prompt
        medical_emergency_template = "Medical emergency: {emergency_message}. Predicted emotion from the speech: {emotion_prediction}. Stress level based on the speech registered in percentage is around: {stressLevel}. You should sssess this situation as an emergency dispatcher when a distressed caller reports a potential life-threatening emergency."

        if urgency_level == 1:
            medical_emergency_template += " Urgency level is critical. Immediate attention is required."

        prompt = medical_emergency_template.format(emergency_message=text['text'], emotion_prediction=emotion, stressLevel=stress)
        st.text(f"Generated Prompt: {prompt}")
        palm.configure(api_key=f"{GOOGLE_API_KEY}")
        # Generate dispatcher response
        context = """ You are an emergency dispatcher working in a centralized emergency response center. Your primary responsibility is to receive calls from individuals facing urgent situations. You are the first point of contact for those in distress. During the call, you must reassure the caller, offering clear instructions and guidance to keep them safe until help arrives. This may involve providing basic first aid instructions or helping the caller remain calm while waiting for emergency responders. Do not give a response in dialogue format just reply to the query as a dispatcher"""

        response = palm.chat(context=context, messages=prompt)
        st.text(f"Dispatcher Response: {response.last}")

        # Convert text to speech
        # text_to_speech(response.last)

        # Append the results to session state history
        session_state.audio_history.append({"filename": './audios/test2.mp3', "emotion": emotion, "stress": stress, "urgency": urgency_level})

    # Display history
    st.subheader("Audio History")
    for item in session_state.audio_history:
        st.text(f"Filename: {item['filename']}")
        st.text(f"Emotion: {item['emotion']}")
        st.text(f"Stress Level: {item['stress']}")
        st.text(f"Urgency Level: {item['urgency']}")
        st.markdown("---")

if __name__ == "__main__":
    main()