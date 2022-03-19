#! /usr/bin/env python3

import os.path
import sqlite3

from ..models.face import Face
from ..models.box import Box
from ..models.imagefaces import ImageFaces

images_home = '../../AFLW/images'

conn = sqlite3.connect('../src/aflw.sqlite')
c = conn.cursor()

query_string = "SELECT image_id, filepath, Faces.face_id, x, y, w, h FROM FaceImages, Faces, FaceRect WHERE FaceImages.file_id = Faces.file_id AND Faces.face_id = FaceRect.face_id"
box_counter = 0

def load_faces(limit = None):
    global box_counter
    
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
            
            face = Face(face_id, Box(box_id = box_counter, x1 = face_x, y1 = face_y, w = face_w, h = face_h))
            box_counter += 1
            
            if image_id in image_data_dict:
                image_data_dict[image_id].add_face(face)
            else:
                img_face = ImageFaces(image_id, input_path)
                img_face.add_face(face)
                
                image_data_dict[image_id] = img_face

    return image_data_dict.values()