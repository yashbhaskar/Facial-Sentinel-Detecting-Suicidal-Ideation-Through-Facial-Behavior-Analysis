import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
from twilio.rest import Client
from collections import deque
from time import time

# Function to send message using Twilio
def send_message():
    account_sid = ' ' # enter account sid
    auth_token = ' '  # enter twilio auth token
    client = Client(account_sid, auth_token)
    
    message = client.messages.create(
        from_='whatsapp:+14155238886',
        body='I detected a depressed face',
        to='whatsapp:+ ' # enter whatsapp number
    )
    print(message.sid)

# Function to evaluate emotions and send message if conditions met
def evaluate_emotions(emotion_counts):
    total_emotions = sum(emotion_counts.values())
    sad_neutral_percentage = (emotion_counts.get('sad', 0) + emotion_counts.get('neutral', 0)) / total_emotions * 100
    
    if sad_neutral_percentage >= 70:
        send_message()

# Load cascade classifier for face detection
faceCascade = cv2.CascadeClassifier("C:\\Users\\ybbha\\Downloads\\Facial_Emotion_Recognition_(Yash_Bhaskar)\\1. Implement CNN and RNN Models\\Facial_emotion_recognition_using_Keras-master\\haarcascade_frontalface_alt2.xml")

# Load pre-trained Keras model for emotion recognition
model = load_model("C:\\Users\\ybbha\\Downloads\\Facial_Emotion_Recognition_(Yash_Bhaskar)\\1. Implement CNN and RNN Models\\Facial_emotion_recognition_using_Keras-master\\keras_model\\model_5-49-0.62.hdf5")
target = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Initialize variables and settings
font = cv2.FONT_HERSHEY_SIMPLEX
video_capture = cv2.VideoCapture(0)
emotion_window = deque(maxlen=300)  # Keep track of emotions for 5 minutes at 60 fps
emotion_counts = {}

# GPU memory fraction configuration
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

# Main loop
while True:
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 5)
        face_crop = frame[y:y + h, x:x + w]
        face_crop = cv2.resize(face_crop, (48, 48))
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        face_crop = face_crop.astype('float32') / 255
        face_crop = np.asarray(face_crop)
        face_crop = face_crop.reshape(1, 1, face_crop.shape[0], face_crop.shape[1])
        
        # Predict emotion
        emotion_index = np.argmax(model.predict(face_crop))
        result = target[emotion_index]
        cv2.putText(frame, result, (x, y), font, 1, (200, 0, 0), 3, cv2.LINE_AA)
        
        # Update emotion window and counts
        emotion_window.append(result)
        emotion_counts[result] = emotion_counts.get(result, 0) + 1
        
        # Evaluate emotions every 5 minutes
        if len(emotion_window) == 300:
            evaluate_emotions(emotion_counts)
            # Reset emotion counts and window
            emotion_counts = {}
            emotion_window.clear()

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
