'''This module contains all the parameters used in the project.'''

#name of face detector
detector_name = 'opencv' #set opencv, ssd, dlib, mtcnn or retinaface

#name of face recognizer
recognizer_name = 'facenet' #set facenet, vggface, openface, deepface, deepid, dlib, arcface

#db path
db_path = 'dataset/train/pics/' #path where images of candidates are stored

#path to save the uploaded video
upload_folder = 'uploaded_video/'

#path to save the processed video
processed_folder = 'results/'

# confidence threshold
threshold = 0.2

#path to save unknown faces
unknown_faces_path = 'results/unknown_faces/'

