from __future__ import division
from numpy import testing
import torch
import os
import cv2
import numpy as np
from common.utils import BBox,drawLandmark,drawLandmark_multiple
from models.basenet import MobileNet_GDConv
from models.pfld_compressed import PFLDInference
from models.mobilefacenet import MobileFaceNet
from FaceBoxes import FaceBoxes
from Retinaface import Retinaface
from PIL import Image
import matplotlib.pyplot as plt
from MTCNN import detect_faces
import time

from utils.align_trans import get_reference_facial_points, warp_and_crop_face

mean = np.asarray([ 0.485, 0.456, 0.406 ])
std = np.asarray([ 0.229, 0.224, 0.225 ])
map_location = 'cpu'

def load_model(model_name):
    if model_name=='MobileNet':
        model = MobileNet_GDConv(136)
        model = torch.nn.DataParallel(model)
        # download model from https://drive.google.com/file/d/1Le5UdpMkKOTRr1sTp4lwkw8263sbgdSe/view?usp=sharing
        checkpoint = torch.load('checkpoint/mobilenet_224_model_best_gdconv_external.pth.tar', map_location=map_location)
        print('Use MobileNet as backbone')
    elif model_name=='PFLD':
        model = PFLDInference() 
        # download from https://drive.google.com/file/d/1gjgtm6qaBQJ_EY7lQfQj3EuMJCVg9lVu/view?usp=sharing
        checkpoint = torch.load('checkpoint/pfld_model_best.pth.tar', map_location=map_location)
        print('Use PFLD as backbone') 
        # download from https://drive.google.com/file/d/1T8J73UTcB25BEJ_ObAJczCkyGKW5VaeY/view?usp=sharing
    elif model_name =='MobileFaceNet':
        model = MobileFaceNet([112, 112],136)   
        checkpoint = torch.load('checkpoint/mobilefacenet_model_best.pth.tar', map_location=map_location)      
        print('Use MobileFaceNet as backbone')         
    else:
        print('Error: not suppored backbone')    
    model.load_state_dict(checkpoint['state_dict'])
    return model

def detect_landmarks(model_name,detector,imgname):
    model = load_model(model_name)
    model.eval()

    output_size = 224
    
    if model_name != 'MobileNet':
        output_size= 112
    
    img = cv2.imread(imgname)
    height,width,_ = img.shape

    if detector == 'MTCNN':
        image = Image.open(imgname)
        faces,_ = detect_faces(image)
    elif detector == 'FaceBoxes':
        face_boxes = FaceBoxes()
        faces = face_boxes(img)
    elif detector == 'Retinalface':
        retinaface= Retinaface.Retinaface()    
        faces = retinaface(img)
    else:
        print('Error: not suppored detector')        
    ratio=0
    if len(faces)==0:
        print('NO face is detected!')

    res = []
    for face in faces:
        if face[4] < 0.9:
            continue
        x1=face[0]
        y1=face[1]
        x2=face[2]
        y2=face[3]
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        size = int(min([w, h])*1.2)
        cx = x1 + w//2
        cy = y1 + h//2
        x1 = cx - size//2
        x2 = x1 + size
        y1 = cy - size//2
        y2 = y1 + size

        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)
        new_bbox = list(map(int, [x1, x2, y1, y2]))
        new_bbox = BBox(new_bbox)
        cropped=img[new_bbox.top:new_bbox.bottom,new_bbox.left:new_bbox.right]

        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)            
        cropped_face = cv2.resize(cropped, (output_size, output_size))
        if cropped_face.shape[0]<=0 or cropped_face.shape[1]<=0:
            continue
        temp_face = cropped_face.copy()
        temp_face = temp_face/255.0

        if model_name == 'MobileNet':
            temp_face = (temp_face-mean)/std
        temp_face = temp_face.transpose((2, 0, 1))
        temp_face = temp_face.reshape((1,) + temp_face.shape)
        input = torch.from_numpy(temp_face).float()
        input= torch.autograd.Variable(input)
        start = time.time()
        if model_name == 'MobileFaceNet':
            landmarks = model(input)[0].cpu().data.numpy()
        else:
            landmarks = model(input).cpu().data.numpy()
        end = time.time()            
        print('Time: {:.6f}s.'.format(end - start))
        landmarks = landmarks.reshape(-1,2)
        landmarks = new_bbox.reprojectLandmark(landmarks)
        
        for landmark in landmarks:
            cv2.circle(img,(int(landmark[0]),int(landmark[1])),2,(0,255,0),1)
        cv2.imshow("landmark",img)
        cv2.waitKey(1)
        res.append(landmarks)
    return res

class landmark_detector:
    def __init__(self,model_name):
        self.model = load_model(model_name)
        self.model.eval()
        self.model_name = model_name
    def getfacebb(self,imgname):
        img = cv2.imread(imgname)
        width,height,_=img.shape
        retinaface= Retinaface.Retinaface()    
        faces = retinaface(img)
        bbox = []
        for face in faces:
            x1=face[0]
            y1=face[1]
            x2=face[2]
            y2=face[3]
            set_pts = [(x1,y1),(x2,y2)]
            bbox.append(set_pts)

        return bbox
    def detect_landmarks(self,detector,imgname):
        output_size = 224
    
        if self.model_name != 'MobileNet':
            output_size= 112
        
        img = cv2.imread(imgname)
        height,width,_ = img.shape

        if detector == 'MTCNN':
            image = Image.open(imgname)
            faces,_ = detect_faces(image)
        elif detector == 'FaceBoxes':
            face_boxes = FaceBoxes()
            faces = face_boxes(img)
        elif detector == 'Retinalface':
            retinaface= Retinaface.Retinaface()    
            faces = retinaface(img)
        else:
            print('Error: not suppored detector')        
        ratio=0
        if len(faces)==0:
            print('NO face is detected!')

        res = []
        for face in faces:
            if face[4] < 0.9:
                continue
            x1=face[0]
            y1=face[1]
            x2=face[2]
            y2=face[3]
            w = x2 - x1 + 1
            h = y2 - y1 + 1
            size = int(min([w, h])*1.2)
            cx = x1 + w//2
            cy = y1 + h//2
            x1 = cx - size//2
            x2 = x1 + size
            y1 = cy - size//2
            y2 = y1 + size

            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)
            new_bbox = list(map(int, [x1, x2, y1, y2]))
            new_bbox = BBox(new_bbox)
            cropped=img[new_bbox.top:new_bbox.bottom,new_bbox.left:new_bbox.right]

            if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                cropped = cv2.copyMakeBorder(cropped, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, 0)   
            cropped_face = cv2.resize(cropped, (output_size, output_size))
            if cropped_face.shape[0]<=0 or cropped_face.shape[1]<=0:
                continue
            temp_face = cropped_face.copy()
            temp_face = temp_face/255.0

            if self.model_name == 'MobileNet':
                temp_face = (temp_face-mean)/std
            temp_face = temp_face.transpose((2, 0, 1))
            temp_face = temp_face.reshape((1,) + temp_face.shape)
            input = torch.from_numpy(temp_face).float()
            input= torch.autograd.Variable(input)
            start = time.time()
            if self.model_name == 'MobileFaceNet':
                landmarks = self.model(input)[0].cpu().data.numpy()
            else:
                landmarks = self.model(input).cpu().data.numpy()
            end = time.time()            
            print('Time: {:.6f}s.'.format(end - start))
            landmarks = landmarks.reshape(-1,2)
            landmarks = new_bbox.reprojectLandmark(landmarks)
            
            for landmark in landmarks:
                cv2.circle(img,(int(landmark[0]),int(landmark[1])),2,(0,255,0),1)

            cv2.rectangle(img,(new_bbox.left,new_bbox.top),(new_bbox.right,new_bbox.bottom),(0,255,0),1)
            '''cv2.imshow("landmark",img)
            cv2.waitKey(1)'''
            res.append(landmarks)
        return np.array(res[0]).reshape(68,2)


if __name__ == '__main__':
    dir = "EP03_2"
    MobileFaceNet_detector = landmark_detector('MobileFaceNet')
    Mobilenet_detector = landmark_detector('MobileNet')
    PFLD_detector = landmark_detector('PFLD')
    landmarks = MobileFaceNet_detector.detect_landmarks('Retinalface',dir+"\\"+dir+"_1.jpg")
    img = cv2.imread(dir+"\\"+dir+"_1.jpg")
    for landmakr in landmarks:
        x1,y1 = int(landmakr[0]),int(landmakr[1])
        print(x1,y1)
        cv2.circle(img,(x1,y1),radius=2,color=(0,255,0),thickness=1)
    cv2.imshow("img",img)
    cv2.waitKey(0)
    print(landmarks)