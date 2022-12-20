'''
A flask app to upload video to the server, process it and return the result
'''
import os
import sys
import time
import json
import base64
import cv2

from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename

from deepface_video import build_detector_model, face_recognition_single_frame

#Set the path of uploaded video
app_uploaded_video = Flask(__name__)
app_uploaded_video.config['UPLOAD_FOLDER'] = 'uploaded_video'


#Set parameters for face recognition
detector_name = 'opencv'
detector_backend = build_detector_model(detector_name)
db_path = 'dataset/train/pics/' #path where images of candidates are stored


#flask function to render the html page
@app_uploaded_video.route('/')
def index():
    return render_template('upload_video.html')


#flask function to upload video
@app_uploaded_video.route('/upload_video', methods=['POST'])
def upload_video():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file part'})
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No selected file'})
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app_uploaded_video.config['UPLOAD_FOLDER'], filename))
            return jsonify({'status': 'success', 'message': 'File uploaded successfully'})


# if frame is wrongly rotated then rotate it
def rotate_frame(frame):
    '''Rotate frame if it is wrongly rotated.
    '''
    #check if frame is rotated
    if frame.shape[0] > frame.shape[1]:
        #rotate frame
        #rotate 180 degrees clockwise

        frame = cv2.rotate(frame, cv2.ROTATE_180)

    return frame


#flask function to process video
def process_video():
    '''Perform facial recognition on uploaded video.
    '''
    cnt = 0
    # Capture video stream
    vid_capture = cv2.VideoCapture('dataset/train/videos/Abesit_Final_Year_Non_Stabalized/video_5.mp4')

    #set fps
    vid_capture.set(cv2.CAP_PROP_FPS, 5)
    
    while True:

        # Capture frame-by-frame
        ret, frame = vid_capture.read()

        #check if frame is empty
        if ret == False:
            break

        # Print shape of frame
        print(frame.shape)

        #rotate frame if it is wrongly rotated
        frame = rotate_frame(frame)

        # Perform facial recognition on frame
        processed_frame = face_recognition_single_frame(frame, detector_backend, detector_name, db_path)

        # Display the resulting frame
        cv2.imshow('frame', processed_frame)

        # Press 'q' to quit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        print(cnt)
        cnt+=1

    # Release the video capture object
    vid_capture.release()
    cv2.destroyAllWindows()


#flask function to return result
def return_result():
    pass


if __name__ == '__main__':
    #app_uploaded_video.run(host='0.0.0.0', port=9999, debug=True)
    process_video()