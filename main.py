from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.websockets import WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import cv2
import numpy as np
import logging
import librosa
from app.model import Wav2LipModel
import traceback
from app.utils import process_frame, audio_stream

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

# CORS middleware to allow requests from different origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your needs (e.g., specify your front-end origin)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = Wav2LipModel()

html = """
<!DOCTYPE html>
<html>
<head>
    <title>Wav2Lip Real-time</title>
</head>
<body>
    <h1>Wav2Lip Real-time Demo</h1>
    <video id="video" width="640" height="480" autoplay></video>
    <div id="error" style="color: red;"></div>
    <script>
        var ws = new WebSocket("ws://" + location.host + "/ws");
        var video = document.getElementById('video');
        var errorDiv = document.getElementById('error');

        navigator.mediaDevices.getUserMedia({video: true, audio: true})
        .then(function(stream) {
            video.srcObject = stream;
            var canvas = document.createElement('canvas');
            canvas.width = 640;
            canvas.height = 480;
            var context = canvas.getContext('2d');

            function captureFrame() {
                context.drawImage(video, 0, 0, canvas.width, canvas.height);
                canvas.toBlob(function(blob) {
                    if (blob.size > 0) {
                        ws.send(blob);
                    }
                }, 'image/jpeg');
            }

            setInterval(captureFrame, 50);  // Reduced interval for better real-time performance
        })
        .catch(function(err) {
            console.log("An error occurred: " + err);
            errorDiv.textContent = "Error accessing camera/microphone: " + err.message;
        });

        ws.onmessage = function(event) {
            var blob = new Blob([event.data], {type: 'image/jpeg'});
            video.src = URL.createObjectURL(blob);
        };

        ws.onerror = function(error) {
            console.error("WebSocket Error: ", error);
            errorDiv.textContent = "WebSocket Error: " + error.message;
        };

        ws.onclose = function(event) {
            if (event.wasClean) {
                console.log(`Closed cleanly, code=${event.code}, reason=${event.reason}`);
            } else {
                console.error('Connection died');
            }
            errorDiv.textContent = "WebSocket connection closed. Please refresh the page.";
        };
    </script>
</body>
</html>
"""

@app.get("/")
async def get():
    return HTMLResponse(html)

def process_audio(audio):
    # Ensure audio is the correct sample rate and mono
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    
    # Convert to float32 if not already
    audio = audio.astype(np.float32)
    
    # Normalize audio
    audio = audio / np.max(np.abs(audio))
    
    # Generate mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=audio, 
        sr=16000, 
        n_mels=80, 
        n_fft=800, 
        hop_length=200, 
        fmax=8000
    )
    
    # Convert to log scale
    mel = librosa.power_to_db(mel, ref=np.max)
    
    # Normalize mel spectrogram
    mel = (mel - np.min(mel)) / (np.max(mel) - np.min(mel))
    
    # Transpose and add batch dimension
    mel = mel.T[np.newaxis, np.newaxis, ...]
    
    # Ensure we have at least 5 frames (adjust as needed)
    if mel.shape[2] < 5:
        mel = np.pad(mel, ((0, 0), (0, 0), (0, 5 - mel.shape[2]), (0, 0)), mode='constant')
    
    logging.debug(f"Mel spectrogram shape after processing: {mel.shape}")
    
    return mel

def ensure_3_channels(image):
    logging.debug(f"ensure_3_channels input shape: {image.shape}")
    if image.ndim == 2:  # If the image is grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:  # If the image has an alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif image.shape[2] == 3:  # If the image is already BGR
        pass
    else:
        raise ValueError(f"Unexpected number of channels: {image.shape[2]}")
    logging.debug(f"ensure_3_channels output shape: {image.shape}")
    return image

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_bytes()
            logging.debug(f"Received data size: {len(data)}")
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
            logging.debug(f"Decoded frame shape: {frame.shape}, dtype: {frame.dtype}")

            if frame is None or frame.size == 0:
                logging.error("Failed to decode frame or frame is empty")
                continue

            audio = await audio_stream()
            logging.debug(f"Audio shape: {audio.shape}, dtype: {audio.dtype}")

            # Process frame and audio together
            processed_frame = process_frame(frame, audio, model)
            logging.debug(f"Processed frame shape: {processed_frame.shape}, dtype: {processed_frame.dtype}")

            # Ensure the processed frame is in BGR format for encoding
            if processed_frame.ndim == 2:
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
            elif processed_frame.shape[2] == 4:
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_RGBA2BGR)
            
            logging.debug(f"Final processed frame shape: {processed_frame.shape}, dtype: {processed_frame.dtype}")

            _, buffer = cv2.imencode('.jpg', processed_frame)
            await websocket.send_bytes(buffer.tobytes())

    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")
    except Exception as e:
        logging.error(f"Error in websocket_endpoint: {str(e)}")
        logging.error(traceback.format_exc())
        try:
            await websocket.close(code=1011, reason=str(e))
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)