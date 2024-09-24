import cv2
import numpy as np
import librosa
import sounddevice as sd
import logging
import asyncio
import torch

logging.basicConfig(level=logging.DEBUG)

def process_frame(frame, audio, model):
    logging.debug(f"process_frame input frame shape: {frame.shape}, dtype: {frame.dtype}")
    logging.debug(f"process_frame input audio shape: {audio.shape}, dtype: {audio.dtype}")

    if frame is None or audio is None or len(audio) == 0:
        logging.error("Invalid frame or audio data")
        return frame  # Return original frame if there's invalid data

    # Preprocess the frame (face detection, alignment, etc.)
    face = preprocess_face(frame)
    
    # Convert audio to mel spectrogram
    mel = audio_to_mel(audio)
    
    if mel is None:
        logging.error("Mel spectrogram is None")
        return frame  # Return original frame if mel is None
    
    # Ensure frame is 4D: [batch_size, channels, height, width]
    if isinstance(face, np.ndarray):
        if face.ndim == 3:
            face = torch.from_numpy(face).unsqueeze(0).float()  # Add batch dimension
        elif face.ndim == 2:
            face = torch.from_numpy(face).unsqueeze(0).unsqueeze(0).float()  # Add batch and channel dimensions
    elif isinstance(face, torch.Tensor):
        if face.dim() == 3:
            face = face.unsqueeze(0)
        elif face.dim() == 2:
            face = face.unsqueeze(0).unsqueeze(0)
    
    # Ensure audio is 3D: [batch_size, time, freq]
    if isinstance(mel, np.ndarray):
        if mel.ndim == 2:
            mel = torch.from_numpy(mel).unsqueeze(0).float()  # Add batch dimension
    elif isinstance(mel, torch.Tensor):
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)

    # Convert to torch tensors if they aren't already
    face = face.float() if isinstance(face, torch.Tensor) else torch.from_numpy(face).float()
    mel = mel.float() if isinstance(mel, torch.Tensor) else torch.from_numpy(mel).float()

    # Pass to model
    with torch.no_grad():
        try:
            output = model(mel, face)  # Try calling the model directly
        except Exception as e:
            logging.error(f"Error calling model: {str(e)}")
            return frame  # Return original frame if there's an error
    
    # Post-process the output (blend with original frame, etc.)
    result = postprocess_output(output, frame)
    
    logging.debug(f"process_frame output shape: {result.shape}, dtype: {result.dtype}")
    return result

def preprocess_face(frame):
    logging.debug(f"preprocess_face input shape: {frame.shape}, dtype: {frame.dtype}")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Check if the frame is already grayscale
    if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
        gray = frame if len(frame.shape) == 2 else frame[:,:,0]
    else:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Assume only one face for simplicity
        face = frame[y:y+h, x:x+w]
        logging.debug(f"preprocess_face output shape: {face.shape}, dtype: {face.dtype}")
        return face
    logging.warning("No face detected, returning original frame.")
    return frame  # Return original frame if no face is detected

def audio_to_mel(audio):
    if audio is None or len(audio) == 0:
        logging.error("Audio input is empty or None")
        return None
    
    logging.debug(f"audio_to_mel input shape: {audio.shape}, dtype: {audio.dtype}")
    
    # Convert audio to mel spectrogram
    try:
        mel = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=80, n_fft=800, hop_length=200)
        logging.debug(f"Mel spectrogram shape: {mel.shape}, dtype: {mel.dtype}")
        return mel
    except Exception as e:
        logging.error(f"Error in audio_to_mel: {str(e)}")
        return None

def postprocess_output(output, original_frame):
    logging.debug(f"postprocess_output input shape: {output.shape if output is not None else None}, {original_frame.shape}")

    # Blend the output with the original frame
    if output is not None:
        # Implement blending logic here if needed
        return output
    logging.warning("Output is None, returning original frame.")
    return original_frame

async def audio_stream():
    chunk = 1024
    sample_rate = 16000
    duration = 0.1  # 100ms for better capture

    audio_data = []

    def callback(indata, frames, time, status):
        if status:
            logging.error(f"Audio callback status: {status}")
        audio_data.extend(indata[:, 0])

    try:
        with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate, blocksize=chunk):
            logging.info("Starting audio stream...")
            await asyncio.sleep(duration)  # Capture audio for the specified duration
            
        if len(audio_data) == 0:
            logging.error("No audio data captured")
            return np.zeros(chunk, dtype=np.float32)  # Return silent audio if there's an error

        audio_array = np.array(audio_data, dtype=np.float32)
        logging.debug(f"Captured audio shape: {audio_array.shape}, dtype: {audio_array.dtype}")
        return audio_array
    except Exception as e:
        logging.error(f"Error in audio_stream: {str(e)}")
        return np.zeros(chunk, dtype=np.float32)  # Return silent audio if there's an error