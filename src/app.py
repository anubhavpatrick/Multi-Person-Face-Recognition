''' Reference - https://pyshine.com/Online-Video-Processing-From-Client-Camera/
The client camera is accessed using the getUserMedia() method. Then the video is streamed 
to the server using the MediaRecorder() method. The video is then processed using the DeepFace 
library at the server back and the processed frames are sent back.
''' 

from threading import Thread
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time
import io
from PIL import Image
import base64,cv2
import numpy as np

#Reference - https://python-engineio.readthedocs.io/en/latest/intro.html
from engineio.payload import Payload 

# import methods from deepface_video.py
from deepface_video import face_recognition_single_frame, create_detector_embeddings, build_detector_model, plot_detected_faces, face_detection

# import all_faces_recognized from deepface_video.py
from deepface_video import all_faces_recognized

# import methods from generic_utilities.py
from util.generic_utilities import write_to_file

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


global fps,prev_recv_time,cnt,fps_array
fps=3 #30
prev_recv_time = 0
cnt=0
fps_array=[0]
detector_name = "opencv"


def image(data_image):
    global fps,cnt, prev_recv_time,fps_array, detector_name
    recv_time = time.time()
    text  =  'FPS: '+str(fps)
    frame = (readb64(data_image))
    imgencode = frame
    #maintain fps
    if cnt == 5:
        all_faces_recognized.clear()
        imgencode = face_recognition_single_frame(frame, detector_name)
        #create new thread to process the frame
        #t = Thread(target=face_recognition_single_frame, args=(frame))
        #t.start()
        #imgencode = frame
        pass
    if fps > 2:
        #build detector model
        detector = build_detector_model(detector_name)
        # detect all the faces that are present in the frame
        obj = face_detection(frame, detector, detector_name)
        print(f'Num of faces detected: {len(obj)}')
        plot_detected_faces(obj, frame)

        #if all faces are recognized, then display the text
        if all_faces_recognized:
            cv2.putText(frame, str(all_faces_recognized), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)
        imgencode = frame

    imgencode = cv2.imencode('.jpeg', imgencode,[cv2.IMWRITE_JPEG_QUALITY,100])[1]

        # base64 encode

    stringData = base64.b64encode(imgencode).decode('utf-8')
    b64_src = 'data:image/jpeg;base64,'
    stringData = b64_src + stringData

    # emit the frame back
    emit('response_back', stringData)

    fps = 1/(recv_time - prev_recv_time)
    fps_array.append(fps)
    fps = round(moving_average(np.array(fps_array)),1)
    prev_recv_time = recv_time
    #print(fps_array)
    cnt+=1
    if cnt==6: #30:
        fps_array=[fps]
        cnt=0
        # create a separate thread to write the all_faces_recognized to a file
        thread = Thread(target=write_to_file, args=(all_faces_recognized,))
        thread.start()
        # Reset the all_faces_recognized variable
        #all_faces_recognized.clear()
    
    print(f'{text}')
    print(all_faces_recognized)


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

    #Run the app
    socketio.run(app,host='0.0.0.0',port=9999 ,debug=True)