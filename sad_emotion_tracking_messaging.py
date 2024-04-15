import cv2
import numpy as np
from keras.models import load_model
import tensorflow as tf
from twilio.rest import Client

def message():

    # Twilio credentials

    from twilio.rest import Client

    account_sid = ' ' # enter account sid
    auth_token = ' '  # enter auth token of twilio
    client = Client(account_sid, auth_token)

    message = client.messages.create(
    from_='whatsapp:+14155238886',
    body='I detected dispressed face',
    to='whatsapp:+ ' # enter whatsapp number
    )

    print(message.sid)


config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

faceCascade = cv2.CascadeClassifier("C:\\Users\\ybbha\\Downloads\\Facial_Emotion_Recognition_(Yash_Bhaskar)\\1. Implement CNN and RNN Models\\Facial_emotion_recognition_using_Keras-master\\haarcascade_frontalface_alt2.xml")

video_capture = cv2.VideoCapture(0)
model = load_model("C:\\Users\\ybbha\\Downloads\\Facial_Emotion_Recognition_(Yash_Bhaskar)\\1. Implement CNN and RNN Models\\Facial_emotion_recognition_using_Keras-master\\keras_model\\model_5-49-0.62.hdf5")
model.get_config()

target = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
font = cv2.FONT_HERSHEY_SIMPLEX
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
        emotion_index = np.argmax(model.predict(face_crop))
        result = target[emotion_index]
        cv2.putText(frame, result, (x, y), font, 1, (200, 0, 0), 3, cv2.LINE_AA)
        
        # Send notification if sad, neutral and disgust faces is detected
        if result in ['sad', 'neutral']:
            message()

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
