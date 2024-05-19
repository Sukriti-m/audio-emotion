from flask import Flask, request, jsonify, after_this_request
import tensorflow as tf
import numpy as np
import librosa
from keras.initializers import Orthogonal
import io
from werkzeug.utils import secure_filename
from moviepy.editor import AudioFileClip
import tempfile

app = Flask(__name__)

# Middleware to set cross-origin isolation headers
@app.after_request
def add_cors_headers(response):
    response.headers['Cross-Origin-Opener-Policy'] = 'same-origin'
    response.headers['Cross-Origin-Embedder-Policy'] = 'require-corp'
    return response


# Load the model directly
custom_objects = {'Orthogonal': Orthogonal}
model = tf.keras.models.load_model('saved_model/model_audio.h5', custom_objects=custom_objects)


def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    return mfccs


def convert_webm_to_wav(webm_data):
    with tempfile.NamedTemporaryFile(suffix='.webm') as temp_input, tempfile.NamedTemporaryFile(
            suffix='.wav') as temp_output:
        temp_input.write(webm_data)
        temp_input.flush()
        clip = AudioFileClip(temp_input.name)
        clip.write_audiofile(temp_output.name, codec='pcm_s16le')
        temp_output.seek(0)
        return temp_output.read()


@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Server is running successfully'}), 200


@app.route('/predict-emotion', methods=['POST'])
def predict_emotion():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    file = request.files['audio']
    filename = secure_filename(file.filename)

    if not filename.lower().endswith(('.wav', '.mp3', '.webm')):
        return jsonify({'error': 'Unsupported audio format. Please upload a .wav, .mp3, or .webm file.'}), 400

    # Load the audio file from memory
    audio_data = file.read()

    if filename.lower().endswith('.webm'):
        try:
            audio_data = convert_webm_to_wav(audio_data)
        except Exception as e:
            return jsonify({'error': f'Error converting .webm file: {e}'}), 400

    audio_file = io.BytesIO(audio_data)

    try:
        features = extract_features(audio_file)
    except Exception as e:
        return jsonify({'error': f'Error processing audio file: {e}'}), 400

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
