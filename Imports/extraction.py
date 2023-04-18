### ALL THE GET FUNCTION, FACE , FRAME, FEATURES

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm


def get_faces(paths: list, face_amount: int) -> (np.ndarray, list):
    video_array = []
    invalid_indices = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    #model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    for idx, path in enumerate(tqdm(paths)):
        vc = cv2.VideoCapture(path)
        faces = []
        while len(faces) < face_amount:
            ret, frame = vc.read()
            if ret and frame is not None:    
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)            
                if len(face) > 0:
                    x, y, w, h = max(face, key=lambda x: x[2] * x[3])
                    face_img = frame[y:y+h, x:x+w]
                    face_img = cv2.resize(face_img, (224, 224))
                    #face_img = np.expand_dims(face_img, axis=0)
                    #feature = model.predict(face_img)
                    faces.append(face_img)
            else:
                break
        vc.release()
        if len(faces) == face_amount:
            video_array.append(np.array(faces))
        else:
            invalid_indices.append(idx)
    return np.array(video_array), invalid_indices


def get_frames_v1(paths: list, frames_each_video: int, video_amount: int, resolution: tuple) -> list:
    video_array_colors = []
    for idx, path in enumerate(paths): 
        if idx == video_amount:
            break
        vc = cv2.VideoCapture(path)
        frames_to_skip = (int(vc.get(cv2.CAP_PROP_FRAME_COUNT))-5)/frames_each_video
        frames_to_skip = int(round(frames_to_skip,0))
        video = []
        i = 0
        while vc.isOpened():
            i += 1
            ret, frame = vc.read()
            if ret and frame is not None:
                if i % frames_to_skip != 0:
                    continue
                if frame.shape[0] == 1920:
                    frame = frame.transpose(1, 0, 2)                    
                frame = cv2.resize(frame, resolution)
                video.append((cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255))
            else:
                vc.release()
                break
            if len(video) < frames_each_video:        # for å catch vid me for lite frames
                video.append(video[-1])        
        video_array_colors.append(np.array(video))
    return np.array(video_array_colors)


def get_frames_v2(paths: list, frames_each_video: int, video_amount: int):
    video_array_colors = []
    face_regions = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    for idx, path in enumerate(paths): 
        if idx == video_amount:
            break
        vc = cv2.VideoCapture(path)
        frames_to_skip = (int(vc.get(cv2.CAP_PROP_FRAME_COUNT))-5)/frames_each_video
        frames_to_skip = math.floor(frames_to_skip)
        video = []
        i = 0
        while vc.isOpened():
            i += 1
            ret, frame = vc.read()
            if ret and frame is not None:
                if i % frames_to_skip != 0:
                    continue
                frame = cv2.resize(frame, (1080, 720))
                video.append(frame)
            else:
                vc.release()
                break
        video_array_colors.append(np.array(video))
        video_face_regions = []
        for frame in video:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
                face_img = frame[y:y+h, x:x+w]
                video_face_regions.append(face_img)
        if video_face_regions:
            face_regions.append(video_face_regions[0])
        else:
            print("No faces detected in any frame of this video.")
            face_regions.append(np.zeros((224, 224, 3)))  # Add a placeholder image with the same size as the face images

    return np.array(video_array_colors), face_regions


def extract_features(face_regions, model):
    features = []
    for face_img in face_regions:
        if np.count_nonzero(face_img) == 0:  # If the face image is a placeholder (all zeros)
            features.append(np.zeros_like(features[-1]))  # Add zeros as features
        else:
            face_img = cv2.resize(face_img, (224, 224))
            face_img = preprocess_input(face_img)
            face_img = np.expand_dims(face_img, axis=0)
            feature = model.predict(face_img)
            features.append(feature.squeeze())
            
    return features

def extract_features_v2(video_array):
    features = []
    face_regions = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    noFaceFound = 0
    for video in tqdm(video_array):
        video_features = []
        video_face_regions = []
        for frame in video:
            gray_frame = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
                face_img = frame[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (224, 224))
                face_img = preprocess_input(face_img * 255)
                face_img = np.expand_dims(face_img, axis=0)
                feature = model.predict(face_img)
                video_features.append(feature.squeeze())
                video_face_regions.append(frame[y:y+h, x:x+w])
            else:
                continue

        if video_features:
            features.append(np.mean(video_features, axis=0))
            face_regions.append(video_face_regions[0])
        else:
            print("No faces detected in any frame of this video.")
            noFaceFound += 1
            if features:  # Check if the features list is not empty
                features.append(np.zeros_like(features[-1]))  # Add zeros if no faces are detected
                face_regions.append(np.zeros((224, 224, 3)))  # Add a placeholder image with the same size as the face images

    return np.array(features), face_regions
