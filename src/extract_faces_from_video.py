#A python module to capture a video stream, detect faces and store them in folder

import threading
import time
import cv2
from deepface import DeepFace
from deepface.detectors import FaceDetector
import ffmpeg  
import random


def capture_video(video_path):
    '''Capture a video stream and return it.
    '''
    # Capture video from webcam
    video_capture = cv2.VideoCapture(video_path)
    return video_capture
  

def check_rotation(path_video_file):
    # Reference - https://stackoverflow.com/questions/53097092/frame-from-video-is-upside-down-after-extracting
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
    if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
        rotateCode = cv2.ROTATE_90_CLOCKWISE
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
        rotateCode = cv2.ROTATE_180
    elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
        rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE
    return rotateCode


def correct_rotation(frame, rotateCode):  
    #Ref- https://stackoverflow.com/questions/53097092/frame-from-video-is-upside-down-after-extracting
    return cv2.rotate(frame, rotateCode) 


def store_detected_faces(frame, current_time):
    '''Store unknown faces in a folder.
    '''
    #save current frame as unknown face
    #generate a random number
    r = random.randint(0, 1000000)
    cv2.imwrite(f'detected_frame/{current_time}_{r}.jpg', frame)


def face_detection(video_path):
    '''Perform facial recognition on a video stream.
    '''
    v = capture_video(video_path)
     # check if video requires rotation
    rotateCode = check_rotation(video_path)

    while True:

        # Capture frame-by-frame
        ret, frame = v.read()
        if rotateCode is not None:
            frame = correct_rotation(frame, rotateCode)
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Perform facial recognition
        detector_name = "opencv" # "opencv" or "ssd"

        detector = FaceDetector.build_model(detector_name) #set opencv, ssd, dlib, mtcnn or retinaface

        obj = FaceDetector.detect_faces(detector, detector_name, frame, align=True)

        print(obj)
        
        for i in range(len(obj)):
            #current time in hours, minutes and seconds
            current_time = time.strftime("%H:%M:%S", time.localtime())

            #iterate over all faces detected
            threading.Thread(target=store_detected_faces, args=(frame,current_time)).start()
            
            color = (0, 255, 0)

            #draw rectangle on face
            #Reference - https://github.com/serengil/deepface/blob/master/deepface/commons/realtime.py
            x = obj[i][1][0]; y = obj[i][1][1]
            w = obj[i][1][2]; h = obj[i][1][3]
            #cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2) #draw rectangle to main image
           
        # Display the resulting frame
        cv2.imshow('Video', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    '''Main function.
    '''
    #video path
    video_path = "video_2.mp4"

    # Perform facial recognition
    face_detection(video_path)


if __name__ == "__main__":
    main()