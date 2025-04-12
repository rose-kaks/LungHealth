import os
from pydub import AudioSegment
import time

# Set paths
AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"

# Verify paths
if not all(os.path.exists(p) for p in [AudioSegment.converter, AudioSegment.ffprobe]):
    raise RuntimeError(f"FFmpeg or ffprobe not found: ffmpeg={AudioSegment.converter}, ffprobe={AudioSegment.ffprobe}")
print(f"Successfully set ffmpeg: {AudioSegment.converter}")
print(f"Successfully set ffprobe: {AudioSegment.ffprobe}")

# Rest of imports
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
import tempfile
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path='cough_diagnosis.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Constants
MAX_LENGTH = 100
N_MFCC = 20
SAMPLE_RATE = 16000

# Custom temp directory
custom_temp_dir = os.path.join(os.getcwd(), 'temp')
if not os.path.exists(custom_temp_dir):
    os.makedirs(custom_temp_dir)

# Debug: Test write permission
try:
    with open(os.path.join(custom_temp_dir, 'test.txt'), 'w') as f:
        f.write('test')
    os.remove(os.path.join(custom_temp_dir, 'test.txt'))
    print(f"Write test successful in {custom_temp_dir}")
except Exception as e:
    print(f"Write test failed: {str(e)}")

# Debug: Test manual save
try:
    with open(os.path.join(custom_temp_dir, 'test_save.webm'), 'wb') as f:
        f.write(b'test data')
    os.remove(os.path.join(custom_temp_dir, 'test_save.webm'))
    print("Manual save test successful")
except Exception as e:
    print(f"Manual save test failed: {str(e)}")

def preprocess_audio(file_path):
    print(f"Starting preprocessing for file: {file_path}")
    wav_path = None
    try:
        # Create temporary WAV file without automatic deletion
        wav_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=custom_temp_dir)
        wav_path = wav_file.name
        wav_file.close()  # Close the file handle immediately
        print(f"Temporary WAV file created: {wav_path}")

        # Read and convert audio
        print(f"Attempting to load audio from: {file_path}")
        audio = AudioSegment.from_file(file_path)
        print(f"Audio loaded successfully, duration: {len(audio)} ms")
        if len(audio) == 0:
            raise Exception("Empty audio file")

        audio.export(
            wav_path,
            format='wav',
            parameters=['-ac', '1', '-ar', str(SAMPLE_RATE)]
        )
        time.sleep(0.1)  # Wait for file to be released
        print(f"Audio exported to WAV: {wav_path}")

        # Load with librosa
        print(f"Loading with librosa from: {wav_path}")
        audio_data, sr = librosa.load(
            wav_path,
            sr=SAMPLE_RATE,
            mono=True,
            res_type='kaiser_fast'
        )
        print(f"Librosa loaded audio, length: {len(audio_data)} samples")

        if len(audio_data) < 512:
            raise Exception("Audio too short for analysis")

        # Extract MFCCs
        print("Extracting MFCCs...")
        mfccs = librosa.feature.mfcc(
            y=audio_data,
            sr=sr,
            n_mfcc=N_MFCC,
            n_fft=2048,
            hop_length=512,
            fmin=50,
            fmax=8000
        )
        print(f"MFCCs shape: {mfccs.shape}")

        # Pad/trim
        if mfccs.shape[1] > MAX_LENGTH:
            mfccs = mfccs[:, :MAX_LENGTH]
        else:
            if mfccs.shape[1] < 10:
                raise Exception("Insufficient audio features")
            mfccs = np.pad(
                mfccs,
                ((0, 0), (0, MAX_LENGTH - mfccs.shape[1])),
                mode='constant',
                constant_values=0
            )
        print(f"Padded MFCCs shape: {mfccs.shape}")

        # Normalize
        mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
        print("MFCCs normalized")

        return mfccs.T[np.newaxis, ...]

    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise
    
    finally:
        # Clean up temporary WAV file
        if wav_path and os.path.exists(wav_path):
            try:
                os.unlink(wav_path)
                print(f"Deleted temporary WAV file: {wav_path}")
            except Exception as e:
                print(f"Warning: Could not delete temp WAV file {wav_path}: {str(e)}")

@app.route('/')
def index():
    print("Rendering index.html")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio' not in request.files:
        print("No audio file provided in request")
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        print("No file selected")
        return jsonify({'error': 'No file selected'}), 400
    
    # Validate file extension
    allowed_extensions = {'.webm', '.wav', '.mp3', '.ogg', '.m4a'}
    file_ext = os.path.splitext(audio_file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        print(f"Unsupported file type: {file_ext}")
        return jsonify({'error': f'Unsupported file type: {file_ext}'}), 400
    
    # Use context manager for temporary file with manual deletion
    temp_path = None
    try:
        temp_path = tempfile.NamedTemporaryFile(suffix=file_ext, dir=custom_temp_dir, delete=False).name
        print(f"Creating temporary file with suffix {file_ext} at {temp_path}")
        audio_file.save(temp_path)
        print(f"Temporary file saved: {temp_path}")
        
        # Process the audio
        mfcc_input = preprocess_audio(temp_path)
        
        # Model inference
        print("Setting tensor for model inference")
        interpreter.set_tensor(input_details[0]['index'], mfcc_input.astype(np.float32))
        interpreter.invoke()
        probability = interpreter.get_tensor(output_details[0]['index'])[0][0]
        print(f"Model inference completed, probability: {probability}")
        
        threshold = 0.4889
        label = 'Unhealthy' if probability > threshold else 'Healthy'
        print(f"Prediction: {label} with probability {probability}")
        
        return jsonify({
            'label': label,
            'probability': float(probability),
            'status': 'success'
        })
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500
    
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                print(f"Deleted temporary file: {temp_path}")
            except Exception as e:
                print(f"Warning: Could not delete temp file {temp_path}: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)