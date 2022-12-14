# -*- coding: utf-8 -*-
from deepface import DeepFace


# detector_backend = "opencv", "ssd", "dlib", "mtcnn", "retinaface"
# model_name = "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace"
# distance_metric = "cosine", "euclidean", "euclidean_l2"


#face recognition
face_recognition = DeepFace.find(img_path="dataset/train/pics/Drishti/IMG20221013122610.jpg",
                                    db_path="dataset/train/pics/",
                                    detector_backend="mtcnn",
                                    model_name="VGG-Face",
                                    distance_metric="cosine",
                                    enforce_detection=False)

print(face_recognition)

