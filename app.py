from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import librosa
import io
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the model directly
model = tf.keras.models.load_model('saved_model/model_audio.h5')


# model.save('model_audio.h5', save_format='h5')
# model = tf.keras.models.load_model('model_audio.h5')


def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs


@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Server is running successfully'}), 200


@app.route('/predict-emotion', methods=['POST'])
def predict_emotion():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    filename = secure_filename(file.filename)

    # Load the audio file from memory
    audio_data = file.read()
    audio_file = io.BytesIO(audio_data)

    features = extract_features(audio_file)
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=-1)

    prediction = model.predict(features)
    emotion = np.argmax(prediction)

    emotion_map = {
        0: 'Anger',
        1: 'Disgust',
        2: 'Fear',
        3: 'Happiness',
        4: 'Pleasant Surprise',
        5: 'Sadness',
        6: 'Neutral'
    }

    result = emotion_map.get(emotion, 'Unknown')
    return jsonify({'emotion': result})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
