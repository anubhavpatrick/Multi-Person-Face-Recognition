'''A python module to detect faces in a video stream and perform facial recognition on them.
'''

from deepface import DeepFace


DeepFace.stream("train/", enable_face_analysis=False, time_threshold=1)