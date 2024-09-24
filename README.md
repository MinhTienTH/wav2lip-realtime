# Wav2Lip Real-time Demo

## Detailed Installation Guide

### Python Environment Setup

1. **Install Python 3.8 or higher**: Ensure you have Python 3.8 or higher installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

2. **Create a virtual environment**: It's recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment**:
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4. **Install dependencies**: Navigate to the project directory and install the required dependencies using pip.
    ```bash
    pip install -r requirements.txt
    ```

### Dependencies

The main dependencies for this project are:
- FastAPI
- uvicorn
- numpy
- opencv-python
- librosa
- asyncio

Ensure you have these dependencies listed in your `requirements.txt` file.

## Instructions to Run the Code

1. **Start the FastAPI server**:
    ```bash
    uvicorn app.main:app --reload
    ```

2. **Access the application**: Open your web browser and navigate to `http://127.0.0.1:8000/`. You should see the Wav2Lip Real-time Demo interface.

### Flow of the Project

- The FastAPI server serves an HTML page that accesses the user's webcam and microphone.
- The video stream is captured and sent to the server via WebSocket.
- The server processes the video frames and audio, generating a lip-synced video in real-time.
- The processed video is sent back to the client and displayed in the browser.

## Dockerfile

To run the module inside a Docker container, you can use the following Dockerfile.
