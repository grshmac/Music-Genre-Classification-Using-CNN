import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from finalcnn2 import CNNModel 
import librosa
import tensorflow as tf
from tensorflow.image import resize

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#lets initialize and load the trained model
input_shape = (64, 64, 1) 
num_classes = 10
model = CNNModel(input_shape=input_shape, num_classes=num_classes)


save_dir = "C:/Users/ASUS/Desktop/MCG_Project/model_backup_final/saved_model_final"
#loading weights and biases
def load_model_weights(model, save_dir):

    for name, variable in model.weights.items():
        model.weights[name] = tf.convert_to_tensor(np.load(os.path.join(save_dir, f"{name}_weights.npy")))
    for name, variable in model.biases.items():
        model.biases[name] = tf.convert_to_tensor(np.load(os.path.join(save_dir, f"{name}_biases.npy")))
    print("Model weights and biases loaded successfully.")

load_model_weights(model, save_dir)


#preprocessing audio and generating spectrogram for user upload
def load_and_preprocess_file(filepath, target_shape=(64,64)):
    data = []
    audio_data, sample_rate = librosa.load(filepath, sr=None)

    chunk_duration = 4  
    overlap_duration = 2  

    chunk_samples = chunk_duration * sample_rate
    overlap_samples = overlap_duration * sample_rate

    num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    for i in range(num_chunks):
        #we calculate start and end indices of the chunk
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = audio_data[start:end]

        #compute the mel spectrogram for the chunk
        mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
        mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
        data.append(mel_spectrogram)

    return np.array(data)


#defining function for predicting model based on input
def model_prediction(model, X_test):
    #forward pass through the model
    logits = model.forward(X_test, is_training=False)
    probabilities = tf.nn.softmax(logits, axis=1).numpy()

    predicted_categories = np.argmax(probabilities, axis=1)

    unique_elements, counts = np.unique(predicted_categories, return_counts=True)

    #this determines the most frequent predicted category
    max_count = np.max(counts)
    max_elements = unique_elements[counts == max_count]

    return max_elements[0]


#rendering the html homepage    
@app.route('/')
def index():
    return render_template('index.html', predicted_genre=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audioFile' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['audioFile']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and (file.filename.endswith('.wav') or file.filename.endswith('.mp3')):
        temp_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(temp_file_path)

        classes = ['Blues', 'Classical', 'Country', 'Disco', 'Hip-hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
        
        X_test=load_and_preprocess_file(temp_file_path)
        predicted_genre = model_prediction(model, X_test)

        #remove the temporary file
        os.remove(temp_file_path)

        return render_template('index.html', predicted_genre=classes[predicted_genre])
    else:
        return render_template('index.html', error="This is invalid file format. Please upload a .wav or .mp3 file.", predicted_genre=None)


#func to run app
if __name__ == '__main__':
    app.run(debug=True)
