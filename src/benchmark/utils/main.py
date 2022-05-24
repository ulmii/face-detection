#! /usr/bin/env python3

import sys

from ..models.face import Face
from ..models.box import Box
from ..models.imagefaces import ImageFaces
from ..tools import *
from ..models import *

from datetime import datetime
from IPython.display import HTML

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

def run_detection(tsv_handle, samples, detector: Detector, cv2_filter = None, use_width_height = False, display_data = False, display_filter = None, filter_area = None):
    total_data = len(samples)
    print("Running detection")
    for i, sample in enumerate(samples):
        if display_data == False:
            sys.stdout.write('\r')
            j = (i + 1) / total_data
            sys.stdout.write("[%-20s] %d%% [%d/%d]" % ('='*int(20*j), 100*j, i + 1, total_data))
            sys.stdout.flush()

        image_faces = tf_to_image_faces(sample)
        img = image_faces.img
        
        if cv2_filter is not None:
            img = cv2.cvtColor(img, cv2_filter)

        t1_start = perf_counter_ns()
        boxes, confidence = detector.detect(img)
        t1_stop = perf_counter_ns()
        
        if display_filter is not None:
            img = cv2.cvtColor(img, display_filter)

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

        if confidence is not None and len(confidence) == len(boxes_preds):
            for i, b in enumerate(boxes_preds):
                b.set_confidence(confidence[i])

        acc = image_faces.calculate_prediction(boxes_preds, tsv_handle, filter_area)
        pred = Prediction(t1_stop - t1_start, acc)

        predicted = [b.poly.bounds for b in boxes_preds]
        ground_truth = [f.box.poly.bounds for f in image_faces.faces]

        tsv_handle.append([datetime.utcnow().isoformat()] + pred.write() + [len(image_faces.faces), predicted, ground_truth])
        
        if display_data:
            for face in image_faces.faces:
                b = face.box
                if filter_area is None:
                    cv2.rectangle(img, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 0), 2)
                elif b.poly.area > filter_area:
                    cv2.rectangle(img, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 0), 2)

            print(pred.stats())

            plt.imshow(img)
            plt.show()
    
    tsv_handle.append_load(10)

def run_detection_video(samples, detector: Detector, cv2_filter = None, use_width_height = False, display_results = False):
    if display_results:
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

    for i, sample in enumerate(samples):
        sequence = sample['sequences']

        filenames = sequence['image/filename']
        images = sequence['image']
        faces = sequence['face']

        fig = plt.figure()
        frames = []
        intersections = []
        unions = []
        confidences = []
        inference_times = []
        for i in range(len(filenames)):
            tf_obj = {
                'image': images[i],
                'faces': {'bbox': [faces[i]]},
                'image/filename': filenames[i],
            }

            image_faces = tf_to_image_faces(tf_obj)
            img = image_faces.img
            
            if cv2_filter is not None:
                img = cv2.cvtColor(img, cv2_filter)

            t1_start = perf_counter_ns()
            boxes, confidence = detector.detect(img)
            t1_stop = perf_counter_ns()
            
            inference_times.append(t1_stop - t1_start)
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

                        if display_results:
                            cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 7)
                    else:
                        x1 = box[0]
                        y1 = box[1]
                        x2 = box[2]
                        y2 = box[3]

                        boxes_preds.append(Box(box_id = get_box_id(), x1 = x1, y1 = y1, x2 = x2, y2 = y2))

                        if display_results:
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 7)

            if confidence is not None and len(confidence) == len(boxes_preds):
                for i, b in enumerate(boxes_preds):
                    b.set_confidence(confidence[i])

            boxes_preds_len = len(boxes_preds)
            if boxes_preds_len > 0 and len(image_faces.faces):
                sorted_boxes_preds = sorted(boxes_preds, key=lambda item: item.confidence, reverse=True)
                pred_box = sorted_boxes_preds[0]
                gt_box = image_faces.faces[0].box
                intersections.append(pred_box.intersection(gt_box))
                unions.append(pred_box.union(gt_box))
                confidences.append(pred_box.confidence)

            if display_results:
                for face in image_faces.faces:
                    b = face.box
                    cv2.rectangle(image_faces.img, (b.x1, b.y1), (b.x2, b.y2), (0, 255, 0), 2)

                im = plt.imshow(image_faces.img, animated=True)
                frames.append([im])
        
        if display_results: 
            ani = animation.ArtistAnimation(fig, frames, interval=30, blit=True, repeat_delay=0)
            html = HTML(ani.to_jshtml())
            display(html)
            print("Video STT-AP: {0:.2f}".format(np.sum(intersections) / np.sum(unions)))
            print("Mean confidence of all frames: {:.2f}".format(np.mean(confidences)))
            print("Mean inference time of all frames: {:.2f}ms".format(np.mean(inference_times) / 1e+6))
        
        plt.close()