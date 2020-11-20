# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A demo that runs hand tracking and object detection on camera frames using OpenCV. 2 EDGETPU
"""
import argparse
import collections
import common
import cv2
import numpy as np
import os
import math
from PIL import Image
import re
from edgetpu.detection.engine import DetectionEngine

from tracker.centroidtracker import CentroidTracker
from tracker.trackableobject import TrackableObject
from imutils.video import VideoStream

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

import base64
import cv2
import zmq

import time
import svgwrite
import gstreamer
from pose_engine import PoseEngine
import tflite_runtime.interpreter as tflite

#===========streamming======================
context = zmq.Context()
footage_socket = context.socket(zmq.PUB)
footage_socket.connect('tcp://147.46.123.186:4664') 
#===========================================
Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

def shadow_text(cv2_im, x, y, text, font_size=16):
    cv2_im = cv2.putText(cv2_im, text, (x + 1, y + 1),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

def mapcamto2dplane(pts_in):
    # provide points from image 1
    pts_src = np.array([[7, 476], [6, 185], [635, 138],[638, 477], [1, 191]])
    # corresponding points from image 2 (i.e. (154, 174) matches (212, 80))
    pts_dst = np.array([[7, 476], [9, 8], [636, 8],[638, 477], [1, 191]])#np.array([[212, 80],[489, 80],[505, 180],[367, 235], [144,153]])

    # calculate matrix H
    h, status = cv2.findHomography(pts_src, pts_dst)

    # provide a point you wish to map from image 1 to image 2
    #pts_in = np.array([[154, 174]], dtype='float32')
    #pts_in = np.array([pts_in])

    # finally, get the mapping
    pointsOut = cv2.perspectiveTransform(pts_in, h)
    pointsOut = np.array([pointsOut])
    point_out = [b for a in pointsOut for b in a]
    return point_out

def check_distance(x1,y1,x2,y2):
    dist = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    return dist

def avg_fps_counter(window_size):
    window = collections.deque(maxlen=window_size)
    prev = time.monotonic()
    yield 0.0  # First fps value.

    while True:
        curr = time.monotonic()
        window.append(curr - prev)
        prev = curr
        yield len(window) / sum(window)


#==============================
def load_labels(path):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(path, 'r', encoding='utf-8') as f:
       lines = (p.match(line).groups() for line in f.readlines())
       return {int(num): text.strip() for num, text in lines}

class BBox(collections.namedtuple('BBox', ['xmin', 'ymin', 'xmax', 'ymax'])):
    """Bounding box.
    Represents a rectangle which sides are either vertical or horizontal, parallel
    to the x or y axis.
    """
    __slots__ = ()

def get_output(interpreter, score_threshold, top_k, image_scale=1.0):
    """Returns list of detected objects."""
    boxes = common.output_tensor(interpreter, 0)
    class_ids = common.output_tensor(interpreter, 1)
    scores = common.output_tensor(interpreter, 2)
    count = int(common.output_tensor(interpreter, 3))

    def make(i):
        ymin, xmin, ymax, xmax = boxes[i]
        return Object(
            id=int(class_ids[i]),
            score=scores[i],
            bbox=BBox(xmin=np.maximum(0.0, xmin),
                      ymin=np.maximum(0.0, ymin),
                      xmax=np.minimum(1.0, xmax),
                      ymax=np.minimum(1.0, ymax)))

    return [make(i) for i in range(top_k) if scores[i] >= score_threshold]

#===GET POSITION TO MAPPING BETWEEN 2 WINDOW================
posList=[]
def onMouse(event, x,y, flags, param):
    global posList
    if event == cv2.EVENT_LBUTTONDOWN:
        posList.append((x,y))

#===========================================================
def main():
    
    default_model_dir = '../all_models'
    
    default_model = '../all_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
    
    default_labels = '../all_models/coco_labels.txt'
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir,default_labels))
    
    parser.add_argument('--top_k', type=int, default=1,
                        help='number of categories with highest score to display')
    
    parser.add_argument('--camera_idx', type=str, help='Index of which video source to use. ', default = 0)
    
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='classifier score threshold')
    
    args = parser.parse_args()
    #=================================================================
    # initialize the frame dimensions (we'll set them as soon as we read
    # the first frame from the video)
    W = 640
    H = 480
    ct = CentroidTracker()
    trackers = []
    trackableObjects = {}

    # initialize the total number of frames processed thus far, along
    # with the total number of objects that have moved either up or down
    totalFrames = 0
    totalDown = 0
    totalUp = 0
    #===========================================================================
    #print('Loading Handtracking model {} with {} labels.'.format(args.model, args.labels))

    #engine = DetectionEngine(args.model)
    #labels = load_labels(args.labels)
    #=====================================================================
    src_size = (W, H)
    print('Loading Detection model {} with {} labels.'.format(args.model,args.labels))
    engine2 = DetectionEngine(args.model)
    labels2 = load_labels(args.labels)
    cap = cv2.VideoCapture(args.camera_idx)
    while cap.isOpened():
    #while cv2.waitKey(1)<0:
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)
        
        #declare new window for show pose in 2d plane========================
        h_cap, w_cap, _ = cv2_im.shape
        cv2_sodidi = np.zeros((h_cap,w_cap,3), np.uint8)
        #================================================================
        objs = engine2.detect_with_image(pil_im,
                                  threshold=0.2,
                                  keep_aspect_ratio=True,
                                  relative_coord=True,
                                  top_k=3)
        cv2_im,rects = append_objs_to_img(cv2_im, objs, labels2)
        
        
        #==================tracking=================================================
        status = "Waiting"
        cv2.line(cv2_im, (W//2, 0), (W//2, H), (0, 255, 255), 2)
        objects = ct.update(rects)
        
        print(objects)
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                # check to see if the object has been counted or not
                if not to.counted:
                    # if the direction is negative (indicating the object
                    # is moving up) AND the centroid is above the center
                    # line, count the object
                    if direction < 0 and centroid[1] < H // 2:
                        totalUp += 1
                        to.counted = True

                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif direction > 0 and centroid[1] > H // 2:
                        totalDown += 1
                        to.counted = True

            # store the trackable object in our dictionary
            trackableObjects[objectID] = to

            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(cv2_im, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(cv2_im, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        # construct a tuple of information we will be displaying on the
        # frame
        info = [
            ("Up", totalUp),
            ("Down", totalDown),
            ("Status", status),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(cv2_im, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        #======================================================================    
        
        cv2.imshow('frame', cv2_im)
        cv2.imshow('1', cv2_sodidi)
        
        #===========print mouse pos=====================
        cv2.setMouseCallback('frame',onMouse)
        posNp=np.array(posList)
        print(posNp)
        #============streamming to server==============
        img = np.hstack((cv2_im, cv2_sodidi))
        #thay frame = img
        encoded, buffer = cv2.imencode('.jpg', img)
        jpg_as_text = base64.b64encode(buffer)
        footage_socket.send(jpg_as_text)
        #=============================================
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()



def append_objs_to_img(cv2_im, objs, labels):
    height, width, channels = cv2_im.shape
    rects = []
    for obj in objs:
        x0, y0, x1, y1 = obj.bounding_box.flatten().tolist() #list(obj.bbox)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        percent = int(100 * obj.score)
        
        label = '{}% {}'.format(percent, labels.get(obj.label_id, obj.label_id))
        if (labels.get(obj.label_id, obj.label_id)=='person'):
            #=====tracking====================
            box = x0, y0, x1, y1
            rects.append(box)
            print(rects)
            #=========================
            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    
    return cv2_im,rects

 
if __name__ == '__main__':
    main()