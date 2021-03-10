
from cv2 import cv2 
import numpy as np 
from os import listdir       # we use this libaray when we have to fetch data from other directory 
from os.path import isfile, join 


data_path = 'faces/'         # location were data is store 
only_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
# it take the files present in faces folder [ listdir(folder name) if file present in it join that file to f ]

training_data, label = [], []
for i , files in enumerate(only_files):           # enumerate provide interation of given number of to files in a folder 
    image_path = data_path + only_files[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    training_data.append(np.asarray(images, dtype=np.uint8))
    label.append(i)

label = np.asarray(label, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()    # .LinearBinaryPhaseHistogramFace Recognizer      face recognizer

# going to tain the model
model.train(np.asarray(training_data), np.asarray(label))
print("training completed !!!")

face_classifier = cv2.CascadeClassifier('C:/Users/n8793/.conda/pkgs/libopencv-4.0.1-hbb9e17c_0/Library/etc/haarcascades/haarcascade_frontalface_alt.xml')     

def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if faces is():
        return img, []

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,255), 2)

        region_of_interest = img[y:y+h, x:x+w]
        region_of_interest = cv2.resize(region_of_interest, (200,200))

    return img, region_of_interest

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] <  500:
            cofidence = int(100*(1-(result[1])/300))
            display_string = str(cofidence)+ '% confidence that it is user'

        cv2.putText(image, display_string, (100,390), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)

        if cofidence > 75:
            cv2.putText(image, "unlocked", (250,440), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2)
            cv2.imshow('face cropper', image)

        else:
            cv2.putText(image, "locked", (250,240), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (250,122,0), 2)
            cv2.imshow('face cropper', image)

    except:
        cv2.putText(image, "face not found ", (250,240), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (150,122,0), 2)
        cv2.imshow('face cropper', image)
        pass

    if cv2.waitKey(1) == 13:
        break

cam.release(0)
cv2.destroyAllWindows()
