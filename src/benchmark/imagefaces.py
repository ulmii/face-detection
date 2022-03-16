#!/usr/bin/env python3
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from .accuracy import Accuracy

class ImageFaces:
    def __init__(self, image_id, image_path):
        self.image_id = image_id
        self.image_path = image_path
        self.faces = []
    
    def add_face(self, face):
        if face not in self.faces:
            self.faces.append(face)

    def calculate_prediction(self, predicted_boxes: list):
        faces_boxes = {}
        boxes_faces = {}
        sorted_faces_boxes = {}
        for f in self.faces:
            faces_boxes[f.face_id] = []
            for b in predicted_boxes:
                iou = f.box.iou(b)
                faces_boxes[f.face_id].append((b.box_id, iou))
                
                if b.box_id in boxes_faces:
                    boxes_faces[b.box_id].append((f.face_id, iou))
                else:
                    boxes_faces[b.box_id] = [(f.face_id, iou)]
                    
            sorted_faces_boxes[f.face_id] = sorted(faces_boxes[f.face_id], key=lambda item: item[1], reverse=True)
        
        sorted_boxes_faces = {k: sorted(v, key = lambda val: val[1], reverse=True) for k, v in boxes_faces.items()}

        false_positives = 0
        final_ious = []

        for box_id, v in sorted_boxes_faces.items():
            for box_values in v:
                face_id = box_values[0]
                box_iou = box_values[1]
                
                all_boxes_face = sorted_faces_boxes[face_id]
                best_box_id_face = all_boxes_face[0][0]
                
                if box_id == best_box_id_face:
                    del sorted_faces_boxes[face_id]
                    
                    if box_iou > 0.25:
                        final_ious.append(box_iou)
                    else:
                        false_positives += 1
                    break;
                false_positives += 1

        return Accuracy(final_ious, len(final_ious), false_positives, len(self.faces) - len(final_ious))
        
    def __str__(self):
        return "({}, {})".format(self.image_id, str(self.faces))
    
    def __repr__(self):
        return "ImageFaces({}, {})".format(self.image_id, repr(self.faces))