'''A python module to detect faces in a video stream and perform facial recognition on them.
'''

from deepface import DeepFace


DeepFace.stream("dataset/train/pics/", enable_face_analysis=True, time_threshold=1)