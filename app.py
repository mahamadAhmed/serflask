from flask import Flask, request, jsonify
import os
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
import opensmile
from scipy import signal

app = Flask(__name__)

# Load the model and scaler
model = load_model('ensemble')
# model.summary()
mfcc_scaler = joblib.load('mfcc.pkl')
opensmile_scaler = joblib.load('opensmile.pkl')

# Initialize OpenSMILE
openSmile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals
)

def extract_features_opensmile(data, sr):
    features = openSmile.process_signal(data, sr)
    return features.values.flatten()

def extract_features_mfcc(data, sr):
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    return np.ravel(mfcc.T)

def extract_features_spectrogram(data, sr):
    f, t, sxx = signal.spectrogram(data)
    return sxx

def remove_silence(audio_path, top_db=30):
    data, sr = librosa.load(audio_path)
    intervals = librosa.effects.split(data, top_db=top_db)
    non_silent_audio = np.concatenate([data[start:end] for start, end in intervals])
    return non_silent_audio, sr

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        audio_path = os.path.join(os.getcwd(), file.filename)
        file.save(audio_path)
        data, sr = remove_silence(audio_path)
        segment_length = 4 * sr

        features_opensmile = []
        features_mfcc = []
        features_spectrogram = []

        for i in range(0, len(data), segment_length):
            if i + segment_length > len(data):
                segment = data[-segment_length:]
            else:
                segment = data[i:i + segment_length]

            # Extract features Opensmile
            features_opensmile.append(np.array(extract_features_opensmile(segment, sr)))
            # Extract features MFCC
            features_mfcc.append(np.array(extract_features_mfcc(segment, sr)))
            # Extract features spectrogram
            features_spectrogram.append(np.array(extract_features_spectrogram(segment, sr)))

        # Scale features
        features_opensmile = opensmile_scaler.transform(features_opensmile)
        features_mfcc = mfcc_scaler.transform(features_mfcc)

        # Convert to np array
        features_spectrogram = np.array(features_spectrogram)
        features_mfcc = np.array(features_mfcc)
        features_opensmile = np.array(features_opensmile)

        # Predict
        predictions = model.predict([features_mfcc, features_opensmile, features_spectrogram])
        predicted_labels = np.argmax(predictions, axis=1)

        emotion_mapping = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad'}
        predicted_emotions = [emotion_mapping[label] for label in predicted_labels]
        average_prediction = np.mean(predictions, axis=0)
        final_emotion = emotion_mapping[np.argmax(average_prediction)]

        return jsonify({
            'predicted_emotions': predicted_emotions,
            'final_emotion': final_emotion
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
