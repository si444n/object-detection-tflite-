import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import pyttsx3

def mySpeak(message, delay=None):
    engine = pyttsx3.init()
    engine.say('{}'.format(message))
    engine.runAndWait()
    if delay is not None:
        time.sleep(delay)
    else:
        time.sleep(4)

def introSpeak(message):
    engine = pyttsx3.init()
    engine.say('{}'.format(message))
    engine.runAndWait()
    time.sleep(1)

introSpeak("Welcome to Object Detection")
introSpeak("Initializing, please wait")
introSpeak("Object Detection is now ready")

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

        # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
        # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the most recent frame
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.6)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='1280x720')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file
PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

frame_rate_calc = 1
freq = cv2.getTickFrequency()

videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)

cv2.namedWindow('Object detector', cv2.WINDOW_NORMAL)


KNOWN_OBJECT_WIDTH = {
    "person": 113,
    "chair": 140,
    "cat": 65,
    "dog": 105,
    "bottle": 25,
    "couch": 275,
    "potted plant": 60,
    "bed": 500,
    "dining table": 300,
    "toilet": 130,
    "tv": 360,
    "laptop": 100,
    "sink": 100,
    "refrigerator": 240,
    "vase": 55,
    "scissor": 20,
    "stuff toy": 80,
    "bicycle": 280,
    "car": 580,
    "motorcycle": 370,
    "bus": 300,
    "truck": 300,
    "bench": 690,
    "umbrella": 180,
    "skateboard": 500
}


FOCAL_LENGTH_MM = 500  # Focal length of the camera in millimeters

# Define A distance for alerting the user (in meters)
ALERT_DISTANCE_THRESHOLD = 3

# Function to calculate distance based on object size and camera calibration
def calculate_distance(object_width_pixels, KNOWN_OBJECT_WIDTH_MM, focal_length_mm, image_width_pixels):
    # Adjusting for perspective distortion based on image width
    distance_mm = (KNOWN_OBJECT_WIDTH_MM * focal_length_mm) / object_width_pixels
    distance_meters = distance_mm / 100.0  # Convert distance to meters
    # Adjusting for perspective distortion based on image width
    adjusted_distance_meters = distance_meters * (image_width_pixels / 1280.0)  # Assuming 1280px is the original width
    return round(adjusted_distance_meters, 2)  # Round distance to 2 decimal places

# Assuming known parameters (calibrated)
# Width of the object in millimeters

while True:
    t1 = cv2.getTickCount()
    
    frame1 = videostream.read()

    frame = frame1.copy()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    object_detected = False  # Flag to track if any object is detected

    for i in range(len(scores)):
        if ((scores[i] > 0.59) and (scores[i] <= 1.0)):
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

            object_name = labels[int(classes[i])]
            label = '%s' % (object_name)
            score_percentage = scores[i] * 100
            # Calculate object width in pixels
            object_width_pixels = xmax - xmin
            known_object_name = object_name  # Assuming you have the object name stored in a variable

            if known_object_name in KNOWN_OBJECT_WIDTH:
                known_object_width_mm = KNOWN_OBJECT_WIDTH[known_object_name]
            else:
                # Handle cases where the object name is not found in the dictionary
                known_object_width_mm = 0  # Or set a default value

            # Calculate distance using size-based estimation
            distance_meters = calculate_distance(object_width_pixels, known_object_width_mm, FOCAL_LENGTH_MM, imW)
            print(f"Distance to {label}: {distance_meters} meters, {score_percentage} accuracy")
            
            # Check if distance is within alert threshold and object is not already detected
            if distance_meters <= ALERT_DISTANCE_THRESHOLD and not object_detected and distance_meters > 0:
                object_detected = True
                mySpeak(f"{label} detected within {ALERT_DISTANCE_THRESHOLD} meters. Please be cautious.", delay=0.1)
            
            # Check if the center of the detected object is in the middle third of the frame horizontally
                center_x = (xmin + xmax) // 2
                if center_x >= imW // 3 and center_x <= (2 * imW) // 3:
                    mySpeak(f"{label} detected in your front at {distance_meters} meters")
                elif xmin <= imW // 3:  # Within left third
                    mySpeak(f"{label} detected in your left at {distance_meters} meters")
                elif xmax >= (2 * imW) // 3:  # Within right third
                    mySpeak(f"{label} detected in your right at {distance_meters} meters")

    # Check if no object is detected and alert the user
    if not object_detected:
        print("No object detected")
        mySpeak("No obstacles detected. You can proceed to walk.")
        


    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
videostream.stop()

