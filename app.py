from flask import Flask, request, jsonify
import os
import numpy as np
import librosa
import joblib
from tensorflow.keras.models import load_model
import opensmile

app = Flask(__name__)

# Load the model and scaler
model = load_model('res2_model.h5')
scaler = joblib.load('std.pkl')

# Initialize OpenSMILE
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals
)

def extract_features(data, sr):
    features = smile.process_signal(data, sr)
    return features.values.flatten()

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

        # Extract features
        segment_length = 3 * sr
        segments = [data[i:i + segment_length] for i in range(0, len(data), segment_length)]
        features = [extract_features(segment, sr) for segment in segments]
        
        # Scale features and make predictions
        features = scaler.transform(features)
        features = np.expand_dims(features, axis=2)
        predictions = model.predict(features)
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
    app.run(debug=True)
