#! /usr/bin/env python3

import sys
import os.path
import sqlite3

from ..models.face import Face
from ..models.box import Box
from ..models.imagefaces import ImageFaces
from ..tools import *
from ..models import *

from datetime import datetime

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

def run_detection(tsv_handle, samples, detector, use_width_height = False, display_data = False):
    total_data = len(samples)
    print("Running detection, total samples: {}".format(total_data))
    for i, sample in enumerate(samples):
        sys.stdout.write('\r')
        j = (i + 1) / total_data
        sys.stdout.write("[%-20s] %d%% [%d/%d]" % ('='*int(20*j), 100*j, i + 1, total_data))
        sys.stdout.flush()

        image_faces = tf_to_image_faces(sample)
        img = image_faces.img

        t1_start = perf_counter_ns()
        boxes = detector(img)
        t1_stop = perf_counter_ns()

        boxes_preds = []
        if boxes is not None:
            for box in boxes:
                box = [int(b) for b in box]
                if use_width_height:
                    x1 = box[0]
                    y1 = box[1]
                    w = box[2]
                    h = box[3]
                    
                    boxes_preds.append(Box(box_id = get_box_id(), x1 = x1, y1 = y1, w = w, h = h))

                    if display_data:
                        cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 7)
                else:
                    x1 = box[0]
                    y1 = box[1]
                    x2 = box[2]
                    y2 = box[3]
                    
                    boxes_preds.append(Box(box_id = get_box_id(), x1 = x1, y1 = y1, x2 = x2, y2 = y2))

                    if display_data:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 7)

        acc = image_faces.calculate_prediction(boxes_preds)
        pred = Prediction(t1_stop - t1_start, acc)

        predicted = [b.poly.bounds for b in boxes_preds]
        ground_truth = [f.box.poly.bounds for f in image_faces.faces]

        tsv_handle.append([datetime.utcnow().isoformat()] + pred.write() + [len(image_faces.faces), predicted, ground_truth])
        
        if display_data:
            for face in image_faces.faces:
                b = face.box
                cv2.rectangle(img, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 0), 2)

            print(pred.stats())
            plt.imshow(img)
            plt.show()
    
    tsv_handle.append_load(10)