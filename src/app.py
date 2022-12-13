''' Reference - https://pyshine.com/Online-Video-Processing-From-Client-Camera/
The client camera is accessed using the getUserMedia() method. Then the video is streamed 
to the server using the MediaRecorder() method. The video is then processed using the DeepFace 
library at the server back and the processed frames are sent back.
''' 

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import time
import io
from PIL import Image
import base64,cv2
import numpy as np

#Reference - https://python-engineio.readthedocs.io/en/latest/intro.html
from engineio.payload import Payload 

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
    sbuf.write(base64.b64decode(base64_string, ' /'))
    
    # Opens the image from the buffer
    pimg = Image.open(sbuf)

    # Convert the image from RGB to BGR
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)


def moving_average(x):
    '''This function calculates the moving average of the last 30 frames'''
    return np.mean(x)


@socketio.on('catch-frame')
def catch_frame(data):
    emit('response_back', data)  


global fps,prev_recv_time,cnt,fps_array
fps=30
prev_recv_time = 0
cnt=0
fps_array=[0]

@socketio.on('image')
def image(data_image):
    global fps,cnt, prev_recv_time,fps_array
    recv_time = time.time()
    text  =  'FPS: '+str(fps)
    frame = (readb64(data_image))

    imgencode = cv2.imencode('.jpeg', frame,[cv2.IMWRITE_JPEG_QUALITY,40])[1]

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
    if cnt==30:
        fps_array=[fps]
        cnt=0
    

if __name__ == '__main__':
    socketio.run(app,port=9990 ,debug=True)