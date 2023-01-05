#A python module to capture a video stream and perform facial recognition on it.

#import threading
from threading import Thread
import time

import cv2
from deepface import DeepFace
from deepface.detectors import FaceDetector

from util.generic_utilities import write_to_file

#Import parameters
from param import detector_name, db_path, threshold, unknown_faces_path, recognizer_name


#create an empty set containing all faces recognized
faces_recognized_dict = {} #global variable


def format_name(name: str) -> str:
    '''Format name to be displayed on video frame
    '''
    #remove directory path from name
    name = (name.rsplit('/', maxsplit=2))[1]
    return name


def capture_video():
    '''Capture a video stream from webcam and return it.
    '''
    # Capture video from webcam
    video_capture = cv2.VideoCapture(0)#0 is the default camera
    
    #print the current fps
    print(f'Frame rate from webcam: {video_capture.get(cv2.CAP_PROP_FPS)}')

    #reduce fps
    #Next statement not working
    video_capture.set(cv2.CAP_PROP_FPS, 5)

    return video_capture


def store_unknown_faces(frame, current_time):
    '''Store unknown faces in a folder.
    '''
    #save current frame as unknown face
    cv2.imwrite(f'{unknown_faces_path}{current_time}.jpg', frame)


def create_detector_embeddings(custom_detector="mtcnn"):
    '''Create embedding for detector
    '''
    #Following code is required to create embedding for mtcnn
    from deepface.detectors import FaceDetector
    detector_name = custom_detector #set opencv, ssd, dlib, mtcnn or retinaface
    detector = FaceDetector.build_model(detector_name) 
    #read jpeg image as numpy array
    img = cv2.imread("dataset/test.jpg")
    # detect all the faces that are present in the frame
    _ = FaceDetector.detect_faces(detector, detector_name, img , align=False)


def build_detector_model(detector_name='mtcnn'):
    '''Build a detector model.
    '''
    #build detector model
    detector = FaceDetector.build_model(detector_name) 
    return detector


def face_detection(frame, detector, detector_name='mtcnn'):
    '''Detect faces in a frame.
    '''
    # detect all the faces that are present in the frame
    obj = FaceDetector.detect_faces(detector, detector_name, frame, align=False)
    return obj #returns all the detected faces


def plot_detected_faces(obj, frame):
    '''Plot detected faces

    Parameters:
    obj (list): list of detected faces

    Returns:
    frame (numpy array): frame with detected faces
    '''
    #Display time at top middle of frame
    current_time = time.strftime("%H:%M:%S", time.localtime())
    cv2.putText(frame, current_time, (int(frame.shape[1]/2)-30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

    #iterate through all the faces detected
    color = (0, 255, 165) #orange
    for i in range(len(obj)):
        #draw rectangle on face
        #Reference - https://github.com/serengil/deepface/blob/master/deepface/commons/realtime.py
        x = obj[i][1][0]; y = obj[i][1][1]
        w = obj[i][1][2]; h = obj[i][1][3]
        
        # draw bounding box on each face detected
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 1) #draw rectangle on main image

    return frame


cnt = 0 #global variable to keep track of frames
def face_recognition_single_frame(frame, detector_backend, detector_name, db_path="dataset/train/pics/"):
    '''Perform facial recognition on a single frame.
    '''
    #build detector model
    #detector = build_detector_model(detector_name)

    # to keep trak of frames
    global cnt

    #resize frame to 224x224
    #frame = cv2.resize(frame, (224, 224))

    # detect all the faces that are present in the frame
    obj = face_detection(frame, detector_backend, detector_name)

    # face_recognized will store the name of the person
    face_recognized = ''

    # all_faces_recognized is a global var containing all the faces recognized
    global faces_recognized_dict

    # if there is at least one face in the frame
    if len(obj) > 0:
        
        #add the first face to the list
        faces = [obj[0][0]]

        #add the rest of the faces to the list
        for i in range(1, len(obj)):
            faces.append(obj[i][0])

        recognizer_model = recognizer_name #set VGG-Face, Facenet, OpenFace, DeepFace, DeepID, Dlib, ArcFace or AgeNet

        if cnt >= 0:
            # Perform facial recognition by passing detected faces on every fifth processed frame
            # find() will return a dataframe with the name of the person and the similarity distance
            recognized_faces_df = DeepFace.find(faces, model_name=recognizer_model, db_path = db_path, silent=True, enforce_detection=False, prog_bar=False)
                
            #iterate through all the faces detected
            for i in range(len(obj)):
                
                #check if recognized_faces_df is a list containing dataframes
                if isinstance(recognized_faces_df, list):
                    #get the name of the person
                    if recognized_faces_df[i].empty:
                        #no faces detected. Increase count and return
                        cnt+=1
                        return
                    else:
                        face_recognized = recognized_faces_df[i].iloc[0]['identity']
                        face_recognition_distance = recognized_faces_df[i].iloc[0]['VGG-Face_cosine']
                else: #else recognized_faces_df is just a single dataframe
                    if recognized_faces_df.empty:
                        #no faces detected. Increase count and return
                        cnt+=1
                        return
                    else:
                        face_recognized = recognized_faces_df.iloc[0]['identity']
                        face_recognition_distance = recognized_faces_df.iloc[0]['VGG-Face_cosine']

                if face_recognition_distance > 0.2:
                    face_recognized = 'Unknown'
                else:
                    # Get the formatted name to be displayed on video frame
                    # Get the name of the person in correct format
                    face_recognized = format_name(face_recognized)

                #create a dictionary key for each face recognized
                if face_recognized not in faces_recognized_dict.keys():
                    faces_recognized_dict[face_recognized] = [1, cnt] #cnt is the initial frame number the face was recognized
                else:
                    faces_recognized_dict[face_recognized][0] += 1 #update how many times the face was recognized

                #write the name of the person to a file
                t = Thread(target=write_to_file, args=('all_faces_recognized.txt',faces_recognized_dict, cnt))
                t.start()
                
                #create a thread for storing unknown faces
                    #Uncomment the following line to store unknown faces
                    #threading.Thread(target=store_unknown_faces, args=(frame.copy(),current_time)).start()
                    #set bounding box color to red
                    #color = (0, 0, 255)
                #else:
                    #set bounding box color to green
                color = (0, 255, 0)
                    #pass
                
                #draw rectangle on face
                #Reference - https://github.com/serengil/deepface/blob/master/deepface/commons/realtime.py
                x = obj[i][1][0]; y = obj[i][1][1]
                w = obj[i][1][2]; h = obj[i][1][3]
                
                #write name of person above bounding box
                cv2.putText(frame, face_recognized, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        plot_detected_faces(obj, frame)

    # if there is at least one face is recognized
    #if all_faces_recognized:
    #    cv2.putText(frame, str(all_faces_recognized), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    cnt+=1 #increment frame count

    return frame


def face_recognition(video_capture, detector_name, db_path):
    '''Perform facial recognition on a video stream.
    '''

    #build detector model
    detector_backend = build_detector_model(detector_name)

    # Loop through video stream
    while True:

        # Capture frame-by-frame
        ret, frame = video_capture.read() #here ret is a boolean value which is true if the frame is available

        #if frame is available
        if ret:
            # Perform facial recognition on a single frame
            frame = face_recognition_single_frame(frame, detector_backend, detector_name, db_path)
            
            print(type(frame))
            print(frame.shape)

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
    face_recognition(video_capture, detector_name, db_path)

    # Release video capture
    video_capture.release()


if __name__ == "__main__":
    main()