import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from concurrent.futures import ThreadPoolExecutor, as_completed

model_path = "C:\\Users\\kaveenprabodhya\\Desktop\\my-FYP-Research\\project\\models\\model.h5"
model = load_model(model_path)

executor = ThreadPoolExecutor(max_workers=5)

def analyze_emotion_frame_async(frame, user_id, callback):
    """
    Analyzes the emotion of a given frame in an asynchronous manner.
    
    Parameters:
    - frame: The image frame to analyze.
    - callback: The function to call with the analysis result.
    """
    future = executor.submit(analyze_emotion_frame, frame)
    future.add_done_callback(
        lambda x: callback(x.result(), user_id)
    )

def analyze_emotion_frame(frame):
    # Assuming `frame` is an OpenCV image captured from a video stream
    # Resize the frame to match the input shape required by the model, e.g., 224x224
    frame_resized = cv2.resize(frame, (224, 224))

    # Convert color from BGR to RGB (model was trained on RGB images)
    # frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Convert the frame to array format
    img_array = image.img_to_array(frame_resized)

    # Expand dimensions to match the model input shape
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)

    # Preprocess the input image
    img_preprocessed = preprocess_input(img_array_expanded_dims)

    # Make predictions
    predictions = model.predict(img_preprocessed)

    # Map predictions to class labels
    class_labels = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    score = tf.nn.softmax(predictions[0])

    # Retrieve the label with the highest score
    predicted_emotion = class_labels[np.argmax(score)]

    return predicted_emotion

def process_frame_for_motion(gray_frame, reference_frame, threshold=25):
    # Convert frames to grayscale
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray_reference = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)

    # Compute the absolute difference between the current and reference frames
    diff = cv2.absdiff(gray_frame, reference_frame)

    # Apply a threshold to the difference
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Optional: Apply dilation to enhance motion areas
    dilated = cv2.dilate(thresh, None, iterations=2)

    # Find contours to detect significant motion areas
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Determine if significant motion is detected based on contours
    motion_detected = len(contours) > 0

    return motion_detected
