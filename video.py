import numpy as np
import cv2
import time
import tensorflow
import matplotlib.pyplot as plt
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#time.sleep(2)
cap.set(15, -8.0)

face_cascade = cv2.CascadeClassifier('opencv_face_detection/haarcascade_frontalface_alt.xml')

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

#x_test=0
#y_test=0
emotion_facerecognition_model = tensorflow.keras.models.load_model('neuron_models/model_emotion_recognition_jaffe_rafd.h5')

emotion_facerecognition_model.summary()

names = ['ANGRY', 'SAD', 'NEUTRAL', 'DISGUST', 'SURPRISE', 'HAPPY', 'FEAR']

def getLabel(id):
    return names[id]

#loss, acc = emotion_facerecognition_model.evaluate(x_test, y_test)
#print("Restored model, accuracy: {:5.2f}%".format(100*acc))
crop_img=[]
crop_img2=[]
elapsed = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = rescale_frame(frame, percent=70)
    #np.mean(frame, axis=(1, 2))
    # Our operations on the frame come here

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask_cv_cascade = face_cascade.detectMultiScale(frame)
    crop_img3_list = []
    for (column, row, width, height) in mask_cv_cascade:
        crop_img = cv2.rectangle(frame, (column, row), (column + width, row + height), (0, 0, 255), 2)
        crop_img2 = crop_img[row:row + height, column:column + width]
        crop_img2 = cv2.resize(crop_img2, (128, 128))
        #print(crop_img2.shape)
        #camera on color
        #crop_img3 = np.expand_dims(crop_img2, axis=0)
        #IF CAMERA ITS ON GRAYSCALE MODE
        crop_img3 = np.expand_dims(crop_img2, axis=0)
        crop_img3 = np.expand_dims(crop_img3, axis=3)
        crop_img3 = crop_img3 * np.ones((1, 1, 3))

        res = emotion_facerecognition_model.predict_classes(crop_img3)
        cv2.putText(frame, str(getLabel(res[0])), (150, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        cv2.imshow('frame', crop_img)

        #print("end time:", time_var)

    # Display the resulting frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

