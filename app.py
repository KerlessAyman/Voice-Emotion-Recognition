import streamlit as st
import sounddevice as sd
import soundfile as sf
import librosa
import numpy as np
import joblib
import io
import tempfile
import os
from sklearn.svm import SVC

# Function to load the trained model
@st.cache_resource
def load_model():
    model = joblib.load('trained_model.pkl')  # Make sure the file is in the same directory
    return model

# Function to extract MFCC features
def extract_features(audio_data, sample_rate):
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate)
    if mfcc.shape == (20, 104):  # Ensure dimensions match training
        return mfcc.flatten()  # Flatten to 1D array
    else:
        return None

# Function to convert emotion number to English label
def get_emotion_label(emotion_num):
    emotion_dict = {
        0: "neutral",
        1: "calm",
        2: "happy",
        3: "sad",
        4: "angry",
        5: "fearful",
        6: "disgust",
        7: "surprised"
    }
    return emotion_dict.get(emotion_num, "unknown")

# Function to record audio without temporary files
def record_audio():
    sample_rate = 22050
    duration = 3  # seconds
    
    st.info("Recording... Speak now!")
    recording = sd.rec(int(duration * sample_rate),
                      samplerate=sample_rate,
                      channels=1,
                      dtype='float32')
    sd.wait()  # Wait until recording is finished
    st.success("Recording complete!")
    
    # Use BytesIO to avoid file locking issues
    audio_bytes = io.BytesIO()
    sf.write(audio_bytes, recording, sample_rate, format='wav')
    audio_bytes.seek(0)
    
    # Process audio directly from memory
    audio_data, sr = librosa.load(audio_bytes, duration=2.4, offset=0.6)
    
    return audio_data, sr, recording

# Streamlit UI
st.set_page_config(page_title="Voice Emotion Recognition", page_icon="🎤")
st.title("🎤 Voice Emotion Recognition")
st.write("Click the button below to record your voice (3 seconds)")

# Voice recording section
if st.button("🎤 Record Voice"):
    try:
        audio_data, sr, recording = record_audio()
        
        # Display the recorded audio
        st.audio(recording.T, sample_rate=sr)
        st.write("Analyzing emotion...")
        
        # Extract features
        features = extract_features(audio_data, sr)
        
        if features is not None:
            # Load the model
            model = load_model()
            
            # Predict emotion
            emotion_num = model.predict([features])[0]
            emotion = get_emotion_label(emotion_num)
            
            # Display results
            st.success(f"Detected Emotion: **{emotion.upper()}**")
            
            # Show emoji based on emotion
            emotion_icons = {
                "neutral": "😐",
                "calm": "😌",
                "happy": "😊",
                "sad": "😢",
                "angry": "😠",
                "fearful": "😨",
                "disgust": "🤢",
                "surprised": "😲"
            }
            st.markdown(f"## {emotion_icons.get(emotion, '❓')} {emotion.upper()}")
        else:
            st.error("Could not extract features. Please try again.")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure your microphone is working and try again.")

# Sidebar information
st.sidebar.markdown("### How to use:")
st.sidebar.write("1. Click the 'Record Voice' button")
st.sidebar.write("2. Speak clearly for 3 seconds")
st.sidebar.write("3. Wait for the emotion analysis")

st.sidebar.markdown("### Supported Emotions:")
st.sidebar.write("- 😊 Happy")
st.sidebar.write("- 😢 Sad")
st.sidebar.write("- 😠 Angry")
st.sidebar.write("- 😨 Fearful")
st.sidebar.write("- 🤢 Disgust")
st.sidebar.write("- 😲 Surprised")
st.sidebar.write("- 😐 Neutral")
st.sidebar.write("- 😌 Calm")