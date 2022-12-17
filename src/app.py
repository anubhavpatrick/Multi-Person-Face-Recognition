''' Reference - https://pyshine.com/Online-Video-Processing-From-Client-Camera/
The client camera is accessed using the getUserMedia() method. Then the video is streamed 
to the server using the MediaRecorder() method. The video is then processed using the DeepFace 
library at the server back and the processed frames are sent back.
''' 

import os
from threading import Thread
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import io
from PIL import Image
import base64,cv2
import numpy as np

#Reference - https://python-engineio.readthedocs.io/en/latest/intro.html
from engineio.payload import Payload 

# import methods from deepface_video.py
from deepface_video import face_recognition_single_frame, create_detector_embeddings, build_detector_model

# to limit the size of the packets sent to the client
Payload.max_decode_packets = 2048

app = Flask(__name__)

#Reference - https://flask-socketio.readthedocs.io/en/latest/
socketio = SocketIO(app,cors_allowed_origins='*' )


@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')


def readb64(base64_string):
    '''
    This function converts the base64 string to an image'''

    #Reference - https://stackoverflow.com/questions/33754935/convert-base64-string-to-image
    idx = base64_string.find('base64,')
    # Extract the base64 string
    base64_string  = base64_string[idx+7:]

    # Create an in-memory buffer to hold the image
    sbuf = io.BytesIO()

    # Write the image to the buffer
    sbuf.write(base64.b64decode(base64_string))#, ' /'))
    
    # Opens the image from the buffer
    pimg = Image.open(sbuf)

    # Convert the image from RGB to BGR
    return cv2.cvtColor(np.array(pimg),cv2.COLOR_RGB2BGR)


def moving_average(x):
    '''This function calculates the moving average of the last 30 frames'''
    return np.mean(x)


@socketio.on('catch-frame')
def catch_frame(data):
    emit('response_back', data)  


'''global fps,prev_recv_time,cnt,fps_array
fps=3 #30
prev_recv_time = 0
cnt=0'''
fps_array=[0]
detector_name = "ssd"
frames = []
frames_processed = []


def face_recognition():
    '''This function performs the face recognition on the frames received from the client'''

    global detector_name
    detector_backend = build_detector_model(detector_name)
    while True:
        if len(frames) > 0:
            frame = frames.pop(0)
            # Reference -
            try:
                frame_processed = face_recognition_single_frame(frame, detector_backend, detector_name)
                frames_processed.append(frame_processed)
            except Exception as e:
                print(f'Error in face recognition - {e}')
                return

@socketio.on('image')
def image(data_image):
    global fps,cnt, prev_recv_time,fps_array, detector_name
    #recv_time = time.time()
    #fps_formatted  =  'FPS: '+str(fps)
    frame = (readb64(data_image))

    # if frames list is greater than 3, clear the backlog of frames
    if len(frames) > 3:
        frames.clear()
        
    #Add frame to the frames list
    frames.append(frame)

    print(f'Length of raw frames list: {len(frames)}')

    #if frames processed is not empty
    if len(frames_processed) > 0:
        print(f'Length of processed frames list: {len(frames_processed)}')
        frame = frames_processed.pop(0)
        
        imgencode = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY,100])[1]
        stringData = base64.b64encode(imgencode).decode('utf-8')
        b64_src = 'data:image/jpeg;base64,'
        stringData = b64_src + stringData
        # emit the frame back
        emit('response_back', stringData)
        
        '''
        #calculate the fps
        fps = 1/(recv_time - prev_recv_time)
        fps_array.append(fps)
        fps = round(moving_average(np.array(fps_array)),1)
        prev_recv_time = recv_time
        #print(fps_array)'''
        '''cnt+=1
        if cnt==10: #30:
            print(f'Count is {cnt}')
            #write the faces recognized to a file using threads
            t = Thread(target=write_to_file, args=('all_faces_recognized.txt',all_faces_recognized,))
            t.start()
            cnt=0'''
        

#flask function to send file to client
#Reference - https://stackoverflow.com/questions/20646822/how-to-serve-static-files-in-flask
@app.route('/download')
def downloadFile ():
    print("Download file")
    #For windows you need to use drive name [ex: F:/Example.pdf]
    #check if file exists
    if os.path.exists('all_faces_recognized.txt'):
        path = "all_faces_recognized.txt"
        return Response(open(path, 'rb').read(),
                        mimetype='text/plain',
                        headers={"Content-Disposition":"attachment;filename=all_faces_recognized.txt"})
    else:
        return Response("File not found", status=404)


if __name__ == '__main__':
    
    #Code for addressing Tensorflow bug. Uncomment when running on GPUs
    #Reference - https://stackoverflow.com/questions/61021287/tf-2-could-not-create-cudnn-handle-cudnn-status-internal-error
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"-----\n Available GPUs: {len(physical_devices)}\n-----")
    else:
        print("-----\nNot enough GPU hardware devices available. Code will run on CPU\n-----")

    #Following code is required to generate detector embedding
    create_detector_embeddings(detector_name)

    #Delete the file if it exists
    if os.path.exists('all_faces_recognized.txt'):
        os.remove('all_faces_recognized.txt')

    #Start the face recognition thread
    t = Thread(target=face_recognition)
    t.start()

    #Run the app on all ips on port 9999
    socketio.run(app,host='0.0.0.0',port=9999 ,debug=False)