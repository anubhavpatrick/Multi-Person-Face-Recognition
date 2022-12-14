# -*- coding: utf-8 -*-
from deepface import DeepFace
import cv2


# detector_backend = "opencv", "ssd", "dlib", "mtcnn", "retinaface"

#face detection and face alignment
face_detected = DeepFace.detectFace(img_path="dataset/train/pics/Vishesh/IMG20221013121603.jpg",
                                    detector_backend="opencv")
face_detected = cv2.cvtColor(face_detected,cv2.COLOR_BGR2RGB)
cv2.imshow("face_detected",face_detected)
cv2.waitKey(0)