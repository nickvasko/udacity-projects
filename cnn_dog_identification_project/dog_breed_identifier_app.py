import cv2
import sys
import pandas as pd
import numpy as np

from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.models import Sequential

Xception_model = Sequential()
Xception_model.add(GlobalAveragePooling2D(input_shape=(7,7,2048)))
Xception_model.add(Dense(133, activation='softmax'))
Xception_model.load_weights('saved_models/weights.best.Xception.hdf5')

# load list of dog names
dog_names = pd.read_csv('dog_names.csv')
dog_names = list(dog_names['0'])

# returns "True" if face is detected in image stored at img_path
def face_detector(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

from extract_bottleneck_features import *

def Xception_predict_breed(frame, dog_names, Xception_model):
    img = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_LINEAR) 
    x = np.expand_dims(img, axis=0)
    x = x.astype(np.float32)

    bottleneck_feature = extract_Xception(x)
    predicted_vector = Xception_model.predict(bottleneck_feature)
    return dog_names[np.argmax(predicted_vector)]

video_capture = cv2.VideoCapture(1)

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
                
# When everything is done, release the captur
video_capture.release()
cv2.destroyAllWindows()

breed = Xception_predict_breed(frame, dog_names, Xception_model)
cv2.rectangle(frame, (0, 0), (750, 50), (255, 255, 255), -1)
cv2.putText(frame,breed,(10,30), font, 1,(0,0,0),2,cv2.LINE_AA)
cv2.imshow('image', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()