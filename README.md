# Navigation Support System for Visually Disabled Individuals Using Computer Vision

A real-time object detection and audio-based navigation support system developed as my Undergraduate Thesis.  
This project uses computer vision and TensorFlow Lite to detect objects through a webcam and provide spoken navigation cues indicating whether an object is on the left, front, or right of the user.

---

## Features

- Real-time object detection using TensorFlow Lite (TFLite)
- Webcam-based video capture via OpenCV
- Offline text-to-speech feedback using pyttsx3
- Directional navigation cues:
  - Detected in your left
  - Detected in your front
  - Detected in your right
- Configurable confidence threshold and webcam resolution
- Optional Coral Edge TPU acceleration

---

## How the System Works

1. Captures live video frames from the webcam.
2. Processes each frame using a TensorFlow Lite object detection model.
3. Filters detections based on a confidence threshold.
4. Determines the object’s horizontal position within the frame:
   - Left third → Left
   - Middle third → Front
   - Right third → Right
5. Announces detected objects and their position using voice output.

---

## Technologies Used

- Python
- OpenCV
- TensorFlow Lite
- NumPy
- pyttsx3 (Text-to-Speech)

---

## Requirements

- Python 3.x
- OpenCV
- NumPy
- pyttsx3
- One of the following:
  - tensorflow (desktop or laptop)
  - tflite-runtime (Raspberry Pi or lightweight environment)

---

## 1. Environment Setup (Optional)

Create a virtual environment:

    python -m venv venv

Activate the environment:

Windows:
    venv\Scripts\activate

macOS / Linux:
    source venv/bin/activate

---

## 2. Install Dependencies

Install the required Python libraries:

    pip install opencv-python numpy pyttsx3

Then install ONE of the following depending on your setup:

Desktop / Laptop (Recommended):
    pip install tensorflow

Raspberry Pi / Lightweight Environment:
    pip install tflite-runtime

---

## Project Structure

    .
    ├── model/
    │   ├── detect.tflite
    │   └── labelmap.txt
    └── main.py

- detect.tflite – TensorFlow Lite object detection model  
- labelmap.txt – Label map file  
- main.py – Main application script  

---

## Usage

Basic execution:

    python main.py --modeldir model

Custom confidence threshold and resolution:

    python main.py --modeldir model --threshold 0.6 --resolution 1280x720

Use a custom model file:

    python main.py --modeldir model --graph your_model.tflite

Enable Coral Edge TPU acceleration:

    python main.py --modeldir model --edgetpu

Controls:
- Press q to exit the application.

---

## Command-Line Arguments

- --modeldir : Directory containing the TFLite model (required)
- --graph : TFLite model filename (default: detect.tflite)
- --labels : Label map filename (default: labelmap.txt)
- --threshold : Minimum detection confidence (default: 0.6)
- --resolution : Webcam resolution in WxH format
- --edgetpu : Enable Coral Edge TPU acceleration

---

## Limitations

- Voice alerts may repeat for the same object across consecutive frames.
- Directional guidance uses a simple left / middle / right screen division.
- Detection accuracy depends on lighting conditions and camera quality.

---

## Future Improvements

- Object prioritization (e.g., people, vehicles, obstacles)
- Haptic or vibration feedback
- Mobile and embedded system optimization

---

## Author

Christian Villanueva  
Undergraduate Thesis Project  
Navigation Support System for Visually Disabled Individuals Using Computer Vision

---

### This project is intended for academic and educational purposes.  

