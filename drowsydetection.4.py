# Importing OpenCV Library for basic image processing functions
import cv2
# Numpy for array related functions
import numpy as np
# Dlib for deep learning based Modules and face landmark detection
import dlib
#face_utils for basic operations of conversion
from imutils import face_utils
from threading import Thread
import os
from scipy.spatial import distance as dist
import pygame #For playing sound

pygame.mixer.init()
pygame.mixer.music.load('alarm.wav')




#Initializing the camera and taking the instance
cap = cv2.VideoCapture(0)

#Initializing the face detector and landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(mStart, mEnd) = (49, 68)
#status marking for current state
sleep = 0
drowsy = 0
active = 0
status=""
color=(0,0,0)
EYE_AR_THRESH = 0.3
MOUTH_AR_THRESH = 0.9
EYE_AR_CONSEC_FRAMES = 50
alarm_status = False
alarm_status2 = False
saying = False
COUNTER = 0

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)


def mouth_aspect_ratio(mouth):
	# compute the euclidean distances between the two sets of
	# vertical mouth landmarks (x, y)-coordinates
        A = dist.euclidean(mouth[2], mouth[10]) # 51, 59
        B = dist.euclidean(mouth[4], mouth[8]) # 53, 57
        C = dist.euclidean(mouth[3], mouth[9])
        D = dist.euclidean(mouth[0], mouth[6]) # 49, 55
        mar = (A + B +C) / (3 * D)
        return mar
        alarm_status = False

while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    #detected face in faces array
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        eye = final_ear(landmarks)
        ear = eye[0]
        leftEye = eye [1]
        rightEye = eye[2]
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if(ear < EYE_AR_THRESH ):
            COUNTER += 1
            cv2.putText(frame, "COUNTER: {:.2f}".format(COUNTER), (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            #If no. of frames is greater than threshold frames,
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                pygame.mixer.Channel(0).play(pygame.mixer.Sound('alarm.wav'))
                cv2.putText(frame, "You are Drowsy", (150,200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2)
        else:
            pygame.mixer.Channel(0).stop()
            COUNTER = 0

        mouth = landmarks[mStart:mEnd]

        mouthMAR = mouth_aspect_ratio(mouth)
        mar = mouthMAR
		# compute the convex hull for the mouth, then
		# visualize the mouth
        mouthHull = cv2.convexHull(mouth)

        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "MAR: {:.2f}".format(mar), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if mar > MOUTH_AR_THRESH:
                cv2.putText(frame, "Yawning!", (30,60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                pygame.mixer.Channel(1).play(pygame.mixer.Sound('alarm1.wav'))
        else:
            pygame.mixer.Channel(1).stop()    
        
       
        	
        cv2.putText(frame, status, (100,100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color,3)

        for n in range(0, 68):
            (x,y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    
        cv2.imshow("Frame", frame)
        # cv2.imshow("Result of detector", face_frame)
        key = cv2.waitKey(1)
        if key == 27:
      	    break
