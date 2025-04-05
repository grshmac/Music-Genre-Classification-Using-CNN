import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from finalcnn2 import CNNModel 
import librosa
import tensorflow as tf
from tensorflow.image import resize
import matplotlib.pyplot as plt
from werkzeug.utils import secure_filename
import csv
from datetime import datetime
import json
import shutil

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static/processed_images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

#lets initialize and load the trained model
input_shape = (64, 64, 1) 
num_classes = 10
model = CNNModel(input_shape=input_shape, num_classes=num_classes)

save_dir = "C:/Users/ASUS/Desktop/Music-Genre-Classification-Using-CNN/model_backup_final/saved_model_final"
#loading weights and biases
def load_model_weights(model, save_dir):
    for name, variable in model.weights.items():
        model.weights[name] = tf.convert_to_tensor(np.load(os.path.join(save_dir, f"{name}_weights.npy")))
    for name, variable in model.biases.items():
        model.biases[name] = tf.convert_to_tensor(np.load(os.path.join(save_dir, f"{name}_biases.npy")))
    print("Model weights and biases loaded successfully.")

load_model_weights(model, save_dir)

#loading training history
def load_training_history():
    with open("training_history_final.json", "r") as f:
        history = json.load(f)
    return history

#preprocessing audio and generating spectrogram for user upload
#for showing mel matrix only
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

def plot_melspec_chunks(y, sr, output_dir="static/chunk_spectrograms"):
    os.makedirs(output_dir, exist_ok=True)

    chunk_duration = 4
    overlap_duration = 2

    chunk_samples = chunk_duration * sr
    overlap_samples = overlap_duration * sr

    num_chunks = int(np.ceil((len(y) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

    image_paths = []

    for i in range(num_chunks):
        start = i * (chunk_samples - overlap_samples)
        end = start + chunk_samples
        chunk = y[start:end]

        spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sr)
        spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

        plt.figure(figsize=(8, 3))
        librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f"Chunk {i+1}")
        plt.tight_layout()

        image_path = os.path.join(output_dir, f"chunk_{i+1}.png")
        plt.savefig(image_path)
        plt.close()
        image_paths.append(image_path)

    return image_paths

#for showing mel image with features to user
# def load_and_preprocess_file(filepath, target_shape=(64,64)):
#     data = []
#     audio_data, sample_rate = librosa.load(filepath, sr=None)

#     chunk_duration = 4  
#     overlap_duration = 2  

#     chunk_samples = chunk_duration * sample_rate
#     overlap_samples = overlap_duration * sample_rate

#     num_chunks = int(np.ceil((len(audio_data) - chunk_samples) / (chunk_samples - overlap_samples))) + 1

#     for i in range(num_chunks):
#         #we calculate start and end indices of the chunk
#         start = i * (chunk_samples - overlap_samples)
#         end = start + chunk_samples
#         chunk = audio_data[start:end]

#         #compute the mel spectrogram for the chunk
#         mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr=sample_rate)
#         mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
#         data.append(mel_spectrogram)
#     spectrogram_array = np.array(data)

#     # Extract Feature Values
#     features = {
#         "mean": round(np.mean(spectrogram_array), 2),
#         "std": round(np.std(spectrogram_array), 2),
#         "min": round(np.min(spectrogram_array), 2),
#         "max": round(np.max(spectrogram_array), 2)
#     }
#     return spectrogram_array, features

#Generate and save Mel Spectrogram Image
def save_spectrogram_image(spectrogram_array, output_path):
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(spectrogram_array[0, :, :, 0], x_axis="time", y_axis="mel", cmap="viridis")
    plt.colorbar(format='%+2.0f dB')
    plt.title("Preprocessed Mel Spectrogram")
    plt.savefig(output_path)
    plt.close()

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

# #For showing mel matrix only
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

        #log the history
        history_filename = 'history.csv'
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(history_filename, mode='a', newline='') as hfile:
            writer = csv.writer(hfile)
            writer.writerow([file.filename, classes[predicted_genre], timestamp])

        #convert numpy array to a string for easy embedding in HTML
        matrix_str = np.array_str(X_test)  #convert array to a string representation

        #remove the temporary file
        os.remove(temp_file_path)

        return render_template('index.html', predicted_genre=classes[predicted_genre], matrix_str=matrix_str)
    else:
        return render_template('index.html', error="This is invalid file format. Please upload a .wav or .mp3 file.", predicted_genre=None)

@app.route('/about')
def about():
    genre_images = {
        'Blues': '/static/images/blues00000.png',
        'Classical': '/static/images/classical00000.png',
        'Country': '/static/images/country00000.png',
        'Disco': '/static/images/disco00000.png',
        'Hip-hop': '/static/images/hiphop00000.png',
        'Jazz': '/static/images/jazz00000.png',
        'Metal': '/static/images/metal00000.png',
        'Pop': '/static/images/pop00000.png',
        'Reggae': '/static/images/reggae00000.png',
        'Rock': '/static/images/rock00000.png'
    }
    
    music_explanation = """
        Music experts have been trying for a long time to understand sound and what differenciates one song from another. 
        How to visualize sound. What makes a tone different from another.
    """
    cnn_explanation = """
        Our Music Genre Classification model uses a Convolutional Neural Network (CNN) to analyze Mel Spectrogram images 
        extracted from audio files. The CNN model consists of multiple convolutional layers, pooling layers, and fully 
        connected layers that help learn genre-specific audio patterns. The input audio is converted into a Mel Spectrogram 
        before being processed by the CNN for classification.
    """
    dataset_explanation = """
        The data comes from a popular data set called GTZAN. We retrieved it from Kaggle.
        The first component is 100 thirty-second audio files for each genre of music.
        There are 10 genres total in this data set: rock, classical, metal,
        disco, blues, reggae, country, hip-hop, jazz and pop. So, all
        together there are 1000 audio files. The second component
        contains the Mel Spectrogram images for each audio clip.
        A Mel Spectrogram depicts the waveforms of an audio clip.
        An example of this can be seen in below figures for each genre.
        For our implementation we will be using the first component only which are the audio files.
    """    
    return render_template('about.html', genre_images=genre_images, cnn_explanation=cnn_explanation, music_explanation=music_explanation, dataset_explanation=dataset_explanation)

@app.route("/show-chunk-spectrograms")
def show_chunk_spectrograms():
    # visualize_file = "static/audio/Classical_Violino.wav"  # <-- your sample path
    # y, sr = librosa.load(visualize_file, sr=44100)
    # image_paths = plot_melspec_chunks(y, sr)

    chunk_dir = os.path.join(app.root_path, 'static', 'chunk_spectrograms')
    
    #list all chunk images (files that start with chunk_ and end with .png)
    image_paths = [os.path.join('static', 'chunk_spectrograms', f) 
                   for f in os.listdir(chunk_dir) if f.startswith('chunk_') and f.endswith('.png')]
    image_paths.sort()
    #convert to URLs for rendering in HTML
    image_urls = ["/" + path for path in image_paths]

    return render_template("chunk_spectrograms.html", image_urls=image_urls)

@app.route('/history')
def history():
    history_filename = 'history.csv'
    history_data = []
    
    #to read the history from the CSV file
    if os.path.exists(history_filename):
        with open(history_filename, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                history_data.append(row)
    
    return render_template('history.html', history_data=history_data)

import numpy as np

@app.route("/show-preprocessed")
def show_preprocessed():
    data = np.load("preprocessed_data2.npy")
    labels = np.load("preprocessed_labels2.npy")

    decoded_labels = np.argmax(labels, axis=1)

    unique_labels = np.unique(decoded_labels)
    genre_sample_data = []
    genre_sample_labels = []

    for label in unique_labels:
        #find the first occurrence of this label
        index = np.where(decoded_labels == label)[0][0]
        genre_sample_data.append(data[index].tolist())
        genre_sample_labels.append(int(label))  

    return render_template("show_preprocessed.html",
                           data_shape=data.shape,
                           label_shape=labels.shape,
                           sample_data=genre_sample_data,
                           sample_labels=genre_sample_labels)

@app.route('/training-history')
def training_history():
    training_history = load_training_history()
    #load precomputed scores
    precision =  0.8593  
    recall = 0.8551
    f1_score_value = 0.8559

    return render_template(
        'training_history.html',
        training_history=training_history,
        precision=precision,
        recall=recall,
        f1_score=f1_score_value
    )

#func to run app
if __name__ == '__main__':
    app.run(debug=True)
