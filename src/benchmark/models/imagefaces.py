#! /usr/bin/env python3

import numpy as np

from .accuracy import Accuracy

class ImageFaces:
    def __init__(self, image_id, image_path, faces = [], img = None):
        self.image_id = image_id
        self.image_path = image_path
        self.img = img
        self.faces = faces
    
    def add_face(self, face):
        if face not in self.faces:
            self.faces.append(face)

    def calculate_prediction(self, predicted_boxes: list, tsv_handle):
        boxes_dict = {}

        for b in predicted_boxes:
            boxes_dict[b.box_id] = b
        

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
        
        all_ious = []
        final_ious = []
        final_boxes = []
        boxes = []
        for box_id, v in sorted_boxes_faces.items():
            for box_values in v:
                face_id = box_values[0]

                if face_id not in sorted_faces_boxes:
                    continue
                
                box_iou = box_values[1]
                all_boxes_face = sorted_faces_boxes[face_id]
                best_box_id_face = all_boxes_face[0][0]
                
                if box_id == best_box_id_face:
                    del sorted_faces_boxes[face_id]
                    all_ious.append(box_iou)
                    
                    tsv_handle.append_ap([boxes_dict[box_id].confidence, box_iou > 0.25, box_iou > 0.50, box_iou > 0.75])

                    if box_iou > 0.25:
                        final_ious.append(box_iou)
                        final_boxes.append(box_id)
                    
                    del boxes_dict[box_id] 
                    break

            boxes.append(box_id)

        for box_id, b in boxes_dict.items():
            all_ious.append(0.0)
            tsv_handle.append_ap([b.confidence, False, False, False])

        return Accuracy(all_ious, len(final_ious), len(np.setdiff1d(boxes, final_boxes)), len(self.faces) - len(final_ious))
        
    def __str__(self):
        return "({}, {})".format(self.image_id, str(self.faces))
    
    def __repr__(self):
        return "ImageFaces({}, {})".format(self.image_id, repr(self.faces))