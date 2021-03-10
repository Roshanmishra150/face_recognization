from cv2 import cv2 
import numpy as np 

#                                     # face recognigation system part 1
# # Notes => this project is made only to capture image  

face_classifier = cv2.CascadeClassifier('C:/Users/n8793/.conda/pkgs/libopencv-4.0.1-hbb9e17c_0/Library/etc/haarcascades/haarcascade_frontalface_alt.xml')     
# to classsify the face of the person 

def face_extrxtor(img):
    cvt_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(cvt_gray, 1.3, 5)     
        # detectMultiScale() [1/3] Detects objects of different sizes in the input image. 
        # The detected objects are returned as a list of rectangles.
        # Matrix of the type CV_8U containing an image where objects are detected.

    if faces is():
        return None

    for (x,y,w,h) in faces:                # if face have value 
        cropped_face = img[y:y+h, x:x+w]   # img [row, col]

    return cropped_face

cam = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cam.read()
    if face_extrxtor(frame) is not None:
        count+=1
        face = cv2.resize(face_extrxtor(frame), (200,200))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        file_name_path = "faces/user"+str(count)+".jpg"
        cv2.imwrite(file_name_path, face)

        cv2.putText(face, str(count), (50,50), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,255,0), 2)
        cv2.imshow('face cropper', face )

    else:
        print("face not found ")
        pass 

    if cv2.waitKey(1) == 13 or count == 100:
        break
    
cam.release(0)
cv2.destroyAllWindows()
print(' collecting samples is  completed ')


