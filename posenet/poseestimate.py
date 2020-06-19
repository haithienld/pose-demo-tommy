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

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk


import time
import svgwrite
import gstreamer
from pose_engine import PoseEngine
import tflite_runtime.interpreter as tflite

Object = collections.namedtuple('Object', ['id', 'score', 'bbox'])

#==============================
EDGES = (
    ('nose', 'left eye'),
    ('nose', 'right eye'),
    ('nose', 'left ear'),
    ('nose', 'right ear'),
    ('left ear', 'left eye'),
    ('right ear', 'right eye'),
    ('left eye', 'right eye'),
    ('left shoulder', 'right shoulder'),
    ('left shoulder', 'left elbow'),
    ('left shoulder', 'left hip'),
    ('right shoulder', 'right elbow'),
    ('right shoulder', 'right hip'),
    ('left elbow', 'left wrist'),
    ('right elbow', 'right wrist'),
    ('left hip', 'right hip'),
    ('left hip', 'left knee'),
    ('right hip', 'right knee'),
    ('left knee', 'left ankle'),
    ('right knee', 'right ankle'),
)

HEADCHECK = ('nose', 'left eye','right eye' ,'left ear', 'right ear')
SHOULDERCHECK = ('left shoulder', 'right shoulder') 
HIPCHECK = ('left hip','right hip')
KNEECHECK = ('left knee','right knee')
ANKLECHECK = ('left ankle','right ankle')

def shadow_text(cv2_im, x, y, text, font_size=16):
    cv2_im = cv2.putText(cv2_im, text, (x + 1, y + 1),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    #dwg.add(dwg.text(text, insert=, fill='black',
    #                 font_size=font_size, style='font-family:sans-serif'))
    #dwg.add(dwg.text(text, insert=(x, y), fill='white',
    #                 font_size=font_size, style='font-family:sans-serif'))

def draw_pose(cv2_im, cv2_sodidi, pose, numobject, src_size, color='yellow', threshold=0.2):
    box_x = 0
    box_y = 0  
    box_w = 641
    box_h = 480
    scale_x, scale_y = src_size[0] / box_w, src_size[1] / box_h
    xys = {}
    #==bien dung de tinh khoang cach giua cac bo phan trong co the ============
    pts_sodidi = []
    headarea={}
    shoulderarea={}
    elbow={}
    lengbackbone=60
    lengleg= 86
    lengface = 30
    #=======================================================
    for label, keypoint in pose.keypoints.items():        
        if keypoint.score < threshold: continue
        # Offset and scale to source coordinate space.
        kp_y = int((keypoint.yx[0] - box_y) * scale_y)
        kp_x = int((keypoint.yx[1] - box_x) * scale_x)
        cv2_im = cv2.putText(cv2_im, str(numobject),(kp_x + 1, kp_y + 1), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        xys[label] = (numobject,kp_x, kp_y)
        
        cv2.circle(cv2_im,(int(kp_x),int(kp_y)),5,(0,255,255),-1)

    #draw pose in 2d plane========================
    checkankle, checkknee,checkhip,checkshoulder,checkhead = False,False, False,False,False
    knee,hip,shoulder,head = {},{},{},{}
    pts_in = np.array([[1.0, 1.0]], dtype='float32')
    for a in ANKLECHECK:
        if a in xys:
            _,x1,y1 = xys[a]
            pts_in = np.array([[x1, y1]], dtype='float32')
        
        else:
            for b in KNEECHECK:
                if b in xys: 
                    checkknee = True
                    knee = xys[b]
            for c in HIPCHECK:
                if c in xys:
                    checkhip = True
                    hip = xys[c]
            for d in SHOULDERCHECK:
                if d in xys:
                    checkshoulder = True
                    shoulder = xys[d]
            for e in HEADCHECK:
                if e in xys:
                    checkhead = True
                    head = xys[e]
            if checkknee == True and checkhip == True:
                _,x1,y1 = knee
                _,x2,y2 = hip 
                leeeng =  check_distance(x1,y1,x2,y2)
                pts_in = np.array([[x1, y1 + leeeng/2]], dtype='float32')
                break
            if checkhip == True and checkshoulder == True:
                _,x1,y1 = shoulder
                _,x2,y2 = hip 
                leeeng =  check_distance(x1,y1,x2,y2)
                pts_in = np.array([[x2, y2 + lengleg*leeeng/lengbackbone]], dtype='float32')
                break
            if checkhead == True and checkshoulder == True:
                _,x1,y1 = shoulder
                _,x2,y2 = head 
                leeeng =  check_distance(x1,y1,x2,y2)
                pts_in = np.array([[x1, y1 + (lengleg+lengbackbone)*leeeng/lengface]], dtype='float32')    
                break
        
    pts_in = np.array([pts_in])        
    pts_out = mapcamto2dplane(pts_in)
    #print(len(pts_out))
    #print(pts_out[0][0,0])
    #print(pts_out[0][0,1])
    pts_sodidi = np.array([numobject,pts_out[0][0,0],pts_out[0][0,1]])
    #cv2_sodidi = cv2.circle(cv2_sodidi,(int(pts_out[0][0,0]),int(pts_out[0][0,1])),5,(0,255,255),-1)
        #=============================================
    return pts_sodidi, xys
    

    '''
    for a, b in EDGES:
        if a not in xys or b not in xys: continue
        anum,ax, ay = xys[a]
        bnum,bx, by = xys[b]
        print(numobject,a,xys[a],b,xys[b])
        cv2.line(cv2_im,(ax, ay), (bx, by),(0,255,255))
    '''

def mapcamto2dplane(pts_in):
    # provide points from image 1
    pts_src = np.array([[154, 174], [702, 349], [702, 572],[1, 572], [1, 191]])
    # corresponding points from image 2 (i.e. (154, 174) matches (212, 80))
    pts_dst = np.array([[154, 174], [702, 349], [702, 572],[1, 572], [1, 191]])#np.array([[212, 80],[489, 80],[505, 180],[367, 235], [144,153]])

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

def main():
    default_model_dir = '../all_models'
    default_model = 'posenet/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite'
    default_labels = 'hand_label.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='.tflite model path',
                        default=os.path.join(default_model_dir,default_model))
    parser.add_argument('--labels', help='label file path',
                        default=os.path.join(default_model_dir, default_labels))
    parser.add_argument('--top_k', type=int, default=1,
                        help='number of categories with highest score to display')
    parser.add_argument('--camera_idx', type=str, help='Index of which video source to use. ', default = 0)
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='classifier score threshold')
    args = parser.parse_args()

    #print('Loading Handtracking model {} with {} labels.'.format(args.model, args.labels))

    #engine = DetectionEngine(args.model)
    #labels = load_labels(args.labels)
    #=====================================================================
    src_size = (640, 480)
    print('Loading Pose model {}'.format(args.model))
    engine = PoseEngine(args.model)
    #engine = PoseEngine('models/mobilenet/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')
    #=====================================================================
    # for detection
    print('Loading Detection model {} with {} labels.'.format('../all_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite', '../all_models/coco_labels.txt'))
    #interpreter2 = common.make_interpreter('../all_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite')
    #interpreter2.allocate_tensors()
    engine2 = DetectionEngine('../all_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite')
    labels2 = load_labels('../all_models/coco_labels.txt')
    cap = cv2.VideoCapture(args.camera_idx)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame

        
        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)
        #declare new window for show pose in 2d plane========================
        h_cap, w_cap, _ = cv2_im.shape
        cv2_sodidi = np.zeros((h_cap,w_cap,3), np.uint8)
        #======================================pose processing=================================
        
        poses, inference_time = engine.DetectPosesInImage(np.uint8(pil_im.resize((641, 481), Image.NEAREST)))
        #print('Posese is',poses)
    
        n = 0
        sum_process_time = 0
        sum_inference_time = 0
        ctr = 0
        fps_counter  = avg_fps_counter(30)
        
        input_shape = engine.get_input_tensor_shape()

        inference_size = (input_shape[2], input_shape[1])


        #print('Shape is',input_shape)
        #print('inference size is:',inference_size)
        start_time = time.monotonic()
        
        end_time = time.monotonic()
        n += 1
        sum_process_time += 1000 * (end_time - start_time)
        sum_inference_time += inference_time

        avg_inference_time = sum_inference_time / n
        text_line = 'PoseNet: %.1fms (%.2f fps) TrueFPS: %.2f' % (
            avg_inference_time, 1000 / avg_inference_time, next(fps_counter)
        )
        
        shadow_text(cv2_im, 10, 20, text_line)
        numobject = 0
        xys={}
        pts_sodidi_arr=[]
        pts_xys_arr=[]
        listwarning=[]
        #draw_pose(cv2_im, poses, dis, src_size)
        for pose in poses:
            '''
        for i in range(len(poses)-1):
            pose = poses[i]
            
            #print(pose.keypoints.items())
            for label, keypoint in pose.keypoints.items():
                #print(label)
                #print(keypoint)
                if keypoint.score < 0.2: continue
                if (label=='nose'):
                    print('yx0,',keypoint.yx)
                    
            for j in range(len(poses)):
                pose1 = poses[j]
                #print(pose.keypoints.items())
                for label, keypoint in pose1.keypoints.items():
                    if keypoint.score < 0.2: continue
                    if (label=='nose'):
                        print('yx1,',keypoint.yx)    
            '''
            pts_sodidi, xys = draw_pose(cv2_im,cv2_sodidi, pose, numobject, src_size)
            #print(pts_sodidi)
            pts_sodidi_arr.append(pts_sodidi)
            pts_xys_arr.append(xys)
            
            numobject += 1
            #print('len coor_av',coor_ave)
            #print(xys,coor_ave)kghkkgkgkgerg.hbjbbsbdbs
        pts_sodidi_arr = np.array([pts_sodidi_arr])
        v2 = [b for a in pts_sodidi_arr for b in a]
        print(v2)
        print(xys)
        print(numobject)
        
        #leng = coor_ave.length
        #print(leng)

        
        
        
        
        #for a in pts_sodidi_arr:
        #    for b in a:
        #        print(b[0])
        
        for i in range(0,len(v2)):
            a,x1,y1 = v2[i]
            for j in range(1,len(v2)):
                if i == j:
                    break
                b,x2,y2 = v2[j]
                distance = check_distance(x1,y1,x2,y2)
                print('distance',distance)
                if distance > 100:
                    cv2_sodidi = cv2.circle(cv2_sodidi,(int(x1),int(y1)),5,(0,0,255),-1)
                    cv2_sodidi = cv2.circle(cv2_sodidi,(int(x2),int(y2)),5,(0,0,255),-1)
                    listwarning.append(i)
                    listwarning.append(j)
                else:   
                    cv2_sodidi = cv2.circle(cv2_sodidi,(int(x1),int(y1)),5,(255,0,0),-1)
                    cv2_sodidi = cv2.circle(cv2_sodidi,(int(x2),int(y2)),5,(255,0,0),-1)
        print('listwarning',listwarning)
        for a, b in EDGES:
            if a not in xys or b not in xys: continue
            num,ax, ay = xys[a]
            num,bx, by = xys[b]
            if num in listwarning:
            #print(numobject,a,xys[a],b,xys[b])
                cv2.line(cv2_im,(ax, ay), (bx, by),(0,0,255))
            else:
                cv2.line(cv2_im,(ax, ay), (bx, by),(255,0,0))
        #==============================================================================================    
        #cv2_im = append_objs_to_img(cv2_im, objs, labels)

        # detection
        #common.set_input(interpreter2, pil_im)
        #interpreter2.invoke()
        #objs = get_output(interpreter2, score_threshold=0.2, top_k=3)
        objs = engine2.detect_with_image(pil_im,
                                  threshold=0.2,
                                  keep_aspect_ratio=True,
                                  relative_coord=True,
                                  top_k=3)
                                

        cv2_im = append_objs_to_img(cv2_im, objs, labels2)
       
        cv2.imshow('frame', cv2_im)
        cv2.imshow('1', cv2_sodidi)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



def append_objs_to_img(cv2_im, objs, labels):
    
    height, width, channels = cv2_im.shape
    for obj in objs:
        x0, y0, x1, y1 = obj.bounding_box.flatten().tolist() #list(obj.bbox)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.label_id, obj.label_id))
        if(labels.get(obj.label_id, obj.label_id)=='person'):
            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im
    
    '''
    height, width, channels = cv2_im.shape

    boxes_ob = []
    confidences = []
    classIDs = []
    for obj in objs:
        x0, y0, x1, y1 = obj.bounding_box.flatten().tolist() #list(obj.bbox)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        percent = int(100 * obj.score)
        confidence = obj.score
        label = '{}% {}'.format(percent, labels.get(obj.label_id, obj.label_id))
        if(labels.get(obj.label_id, obj.label_id)=='person'):
            classIDs.append(0)
            confidences.append(float(confidence))
            boxes_ob.append([x0, y0, x1,y1])
            #cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
            #cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    idxs = cv2.dnn.NMSBoxes(boxes_ob, confidences, 0.5,0.3)
    #print('idxs',idxs)
    #print('classID',classIDs)
    ind = []
    for i in range(0,len(classIDs)):
        if(classIDs[i]==0):
            ind.append(i)
    a = []
    b = []

    if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes_ob[i][0], boxes_ob[i][1])
                (w, h) = (boxes_ob[i][2], boxes_ob[i][3])
                a.append(x)
                b.append(y)
    distance=[] 
    nsd = []
    for i in range(0,len(a)-1):
        for k in range(1,len(a)):
            if(k==i):
                break
            else:
                x_dist = (a[k] - a[i])
                y_dist = (b[k] - b[i])
                d = math.sqrt(x_dist * x_dist + y_dist * y_dist)
                distance.append(d)
                if(d <=1000):
                    nsd.append(i)
                    nsd.append(k)
                nsd = list(dict.fromkeys(nsd))
                print(nsd)
    color = (0, 0, 255) 
    for i in nsd:
        (x, y) = (boxes_ob[i][0], boxes_ob[i][1])
        (w, h) = (boxes_ob[i][2], boxes_ob[i][3])
        cv2_im=cv2.rectangle(cv2_im, (x, y), (x + w, y + h), color, 2)
        text = "Alert"
        cv2_im=cv2.putText(cv2_im, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
    color = (0, 255, 0) 
    if len(idxs) > 0:
        for i in idxs.flatten():
            if (i in nsd):
                break
            else:
                (x, y) = (boxes_ob[i][0], boxes_ob[i][1])
                (w, h) = (boxes_ob[i][2], boxes_ob[i][3])
                cv2_im=cv2.rectangle(cv2_im, (x, y), (x + w, y + h), color, 2)
                text = 'OK'
                cv2_im=cv2.putText(cv2_im, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)   
    
    #cv2.imshow("Social Distancing Detector", image)      
    return cv2_im
    '''

if __name__ == '__main__':
    main()
