import os;
from flask import Flask, request, jsonify, render_template

#folder to save temp, the inp files
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'audioFile' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['audioFile']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and file.filename.endswith('.wav'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

     return jsonify({'error': 'Invalid file format. Please upload a WAV file.'})

def process_file(file_path):
    # Load audio file using librosa
    y, sr = librosa.load(file_path, sr=None)

if __name__ == '__main__':
    app.run(debug=True)