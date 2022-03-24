#! /usr/bin/env python3

import os.path
import sqlite3

from ..models.face import Face
from ..models.box import Box
from ..models.imagefaces import ImageFaces

images_home = '../../AFLW/images'
query_string = "SELECT image_id, filepath, Faces.face_id, x, y, w, h FROM FaceImages, Faces, FaceRect WHERE FaceImages.file_id = Faces.file_id AND Faces.face_id = FaceRect.face_id"

face_id = 0
box_id = 0
image_id = 0

def get_box_id():
    global box_id
    
    box_id += 1
    return box_id

def get_face_id():
    global face_id
    
    face_id += 1
    return face_id

def get_image_id():
    global image_id
    
    image_id += 1
    return image_id

def tf_to_image_faces(tf_obj):
    img = tf_obj['image']
    height, width, channels = img.shape
    
    faces = tf_obj['faces']
    faces_with_boxes = [Face(get_face_id(), Box(box_id = get_box_id, x1 = int(box[1].numpy() * width), y1 = int(box[2].numpy() * height), x2 = int(box[3].numpy() * width), y2 = int(box[0].numpy() * height))) for box in faces['bbox']]
    image_faces = ImageFaces(get_image_id(), tf_obj['image/filename'].numpy().decode("utf-8"), faces_with_boxes, img.numpy())
    
    return image_faces

def load_aflw(limit = None):
    global box_counter

    conn = sqlite3.connect('../src/aflw.sqlite')
    c = conn.cursor()
    
    query = query_string
    if limit is not None:
        query += ' LIMIT {}'.format(limit)

    image_data_dict = {}

    for row in c.execute(query):
        file_path = str(row[1])
        input_path = images_home + '/' + file_path
        
        if(os.path.isfile(input_path) == True):
            image_id = row[0]
            face_id = row[2]
            face_x = row[3]
            face_y = row[4]
            face_w = row[5]
            face_h = row[6]
            
            face = Face(face_id, Box(box_id = box_id, x1 = face_x, y1 = face_y, w = face_w, h = face_h))
            box_id += 1
            
            if image_id in image_data_dict:
                image_data_dict[image_id].add_face(face)
            else:
                img_face = ImageFaces(image_id, input_path)
                img_face.add_face(face)
                
                image_data_dict[image_id] = img_face

    return image_data_dict.values()