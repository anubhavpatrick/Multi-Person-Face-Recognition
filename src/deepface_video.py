#A python module to capture a video stream and perform facial recognition on it.

import threading
import time

import cv2
from deepface import DeepFace
from deepface.detectors import FaceDetector


def format_name(name: str) -> str:
    '''Format name to be displayed on video frame
    '''
    #remove directory path from name
    name = (name.rsplit('/', maxsplit=1))[1]
    return name


def capture_video():
    '''Capture a video stream from webcam and return it.
    '''
    # Capture video from webcam
    video_capture = cv2.VideoCapture(0)
    
    #print the current fps
    print(f'Frame rate from webcam: {video_capture.get(cv2.CAP_PROP_FPS)}')

    #reduce fps
    #Next statement not working
    #video_capture.set(cv2.CAP_PROP_FPS, 5)

    return video_capture


def store_unknown_faces(frame, current_time):
    '''Store unknown faces in a folder.
    '''
    #save current frame as unknown face
    cv2.imwrite(f'unknown_faces/{current_time}.jpg', frame)


def face_recognition(video_capture, db_path):
    '''Perform facial recognition on a video stream.
    '''
    # Loop through video stream
    while True:

        # Capture frame-by-frame
        ret, frame = video_capture.read() #here ret is a boolean value which is true if the frame is available

        # Perform facial recognition
        detector_name = "ssd" # "opencv" or "ssd"

        detector = FaceDetector.build_model(detector_name) #set opencv, ssd, dlib, mtcnn or retinaface

        # detect all the faces that are present in the frame
        obj = FaceDetector.detect_faces(detector, detector_name, frame, align=True)

        # face_recognized will store the name of the person
        face_recognized = ''
        
        # if there is at least one face in the frame
        if len(obj) > 0:
            
            #add the first face to the list
            imgs = [obj[0][0]]

            #add the rest of the faces to the list
            for i in range(1, len(obj)):
                imgs.append(obj[i][0])

            # Perform facial recognition by passing detected faces
            # find() will return a dataframe with the name of the person and the similarity distance
            recognized_faces_df = DeepFace.find(imgs, db_path = db_path, silent=True, enforce_detection=False, prog_bar=False)
        
        #Display time at top middle of screen
        current_time = time.strftime("%H:%M:%S", time.localtime())
        cv2.putText(frame, current_time, (int(frame.shape[1]/2)-30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
            

        #iterate through all the faces detected
        for i in range(len(obj)):
            
            #check if recognized_faces_df is a list containing dataframes
            if isinstance(recognized_faces_df, list):
                #get the name of the person
                face_recognized = recognized_faces_df[i].iloc[0]['identity']
                face_recognition_distance = recognized_faces_df[i].iloc[0]['VGG-Face_cosine']
            else: #else recognized_faces_df is just a single dataframe
                face_recognized = recognized_faces_df.iloc[0]['identity']
                face_recognition_distance = recognized_faces_df.iloc[0]['VGG-Face_cosine']
            
            # Get the formatted name to be displayed on video frame
            face_recognized = format_name(face_recognized)+f' Similarity_Distance:{face_recognition_distance:.3f}'
            
            if face_recognition_distance > 0.2:
                face_recognized = 'Unknown'
                #create a thread for storing unknowdfn faces
                #Uncomment the following line to store unknown faces
                #threading.Thread(target=store_unknown_faces, args=(frame.copy(),current_time)).start()
                #set bounding box color to red
                color = (0, 0, 255)
            else:
                #set bounding box color to green
                color = (0, 255, 0)

            #draw rectangle on face
            #Reference - https://github.com/serengil/deepface/blob/master/deepface/commons/realtime.py
            x = obj[i][1][0]; y = obj[i][1][1]
            w = obj[i][1][2]; h = obj[i][1][3]
            
            # draw bounding box on each face detected
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2) #draw rectangle to main image
            
            #write name of person above bounding box
            cv2.putText(frame, face_recognized, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display the resulting frame
        cv2.imshow('Face Detection', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def main():
    '''Main function.
    '''
    # Capture video stream
    video_capture = capture_video()

    # Perform facial recognition
    face_recognition(video_capture, "dataset/train/")

    # Release video capture
    video_capture.release()


if __name__ == "__main__":
    main()