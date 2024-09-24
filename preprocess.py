from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.websockets import WebSocketDisconnect
import asyncio
import cv2
import numpy as np
import logging
from app.model import Wav2LipModel
from app.utils import process_frame, audio_stream
import librosa
import torch

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()
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

            setInterval(captureFrame, 100);
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

def pad_audio(audio, target_length):
    if audio.shape[0] < target_length:
        padding = target_length - audio.shape[0]
        audio = np.pad(audio, (0, padding), mode='constant')
    return audio

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    audio_task = asyncio.create_task(audio_stream())
    min_audio_length = 2048

    try:
        while True:
            data = await websocket.receive_bytes()
            logging.debug(f"Received data size: {len(data)}")
            frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

            if frame is None or frame.size == 0:
                logging.error("Failed to decode frame or frame is empty")
                continue

            audio = await audio_task
            audio = pad_audio(audio, min_audio_length)

            if audio.shape[0] < min_audio_length:
                logging.warning("Audio input is still too short after padding.")
                continue

            # Ensure audio is in the correct shape for melspectrogram
            mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=16000, n_mels=128, fmax=8000)
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            mel_spectrogram_db = mel_spectrogram_db[np.newaxis, np.newaxis, :, :]  # Shape (1, 1, 128, T)

            processed_frame = process_frame(frame, mel_spectrogram_db, model)

            if processed_frame.ndim == 3:
                processed_frame = np.expand_dims(processed_frame, axis=0)

            _, buffer = cv2.imencode('.jpg', processed_frame)
            await websocket.send_bytes(buffer.tobytes())
    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")
    except Exception as e:
        logging.error(f"Error in websocket_endpoint: {str(e)}")
    finally:
        audio_task.cancel()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
