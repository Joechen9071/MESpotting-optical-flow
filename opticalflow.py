from models.mobilefacenet import MobileFaceNet
import cv2 
import face_detector
import numpy as np
import os
import flow_vis
side_len = 0
lines = []
def draw_flow(img, flow, step=8):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def getmid_pt (pt1,pt2):
    return (round((pt1[0]+pt2[0])/2), round((pt1[1]+pt2[1])/2))

def facial_area_coordinate(pts):
    global side_len
    areas = []
    if side_len == 0:
        side_len = round(abs(pts[48][0] - pts[54][0])/2)
        if side_len % 2 != 0:
            side_len = side_len + 1

    half_len = round(side_len/2)
    
    area_1_pt1 = (pts[36][0] - side_len, pts[36][1] - half_len)
    area_1_pt2 = (pts[36][0],pts[36][1]+ half_len)
    
    min_x_val = min(area_1_pt1[0],area_1_pt2[0])
    min_y_val = min(area_1_pt1[1],area_1_pt2[1])

    max_x_val = max(area_1_pt1[0],area_1_pt2[0])
    max_y_val = max(area_1_pt1[1],area_1_pt2[1])
    
    area_1_pt1 = (min_x_val,min_y_val)
    area_1_pt2 = (max_x_val,max_y_val)

    areas.append((area_1_pt1,area_1_pt2))

    area_3_mid_pt = getmid_pt(pts[20],pts[23])
    area_3_pt1 = (area_3_mid_pt[0] - half_len,area_3_mid_pt[1] - half_len)
    area_3_pt2 = (area_3_mid_pt[0] + half_len, area_3_mid_pt[1] + half_len)

    areas.append((area_3_pt1,area_3_pt2))
    
    area_2_pt1 = (area_3_pt1[0] - side_len, area_3_pt1[1])
    area_2_pt2 = (area_3_pt2[0] - side_len, area_3_pt2[1])

    areas.append((area_2_pt1,area_2_pt2))

    area_4_pt1 = (area_3_pt1[0] + side_len, area_3_pt1[1])
    area_4_pt2 = (area_3_pt2[0] + side_len, area_3_pt2[1])

    areas.append((area_4_pt1,area_4_pt2))

    area_8_pt1 = (pts[48][0] - half_len,pts[48][1] - half_len)
    area_8_pt2 = (pts[48][0] + half_len,pts[48][1] + half_len)

    areas.append((area_8_pt1,area_8_pt2))

    area_9_pt1 = (pts[54][0] - half_len,pts[54][1]-half_len)
    area_9_pt2 = (pts[54][0] + half_len,pts[54][1]+half_len)

    areas.append((area_9_pt1,area_9_pt2))

    area_6_pt1 = (pts[31][0] - side_len, area_8_pt1[1] - side_len)
    area_6_pt2 = (pts[31][0],area_8_pt1[1])
    areas.append((area_6_pt1,area_6_pt2))

    area_7_pt1 = (pts[35][0],area_9_pt1[1] - side_len)
    area_7_pt2 = (pts[35][0] + side_len,area_9_pt1[1])
    areas.append((area_7_pt1,area_7_pt2))

    area_5_pt1 = (pts[45][0],pts[45][1] - half_len)
    area_5_pt2 = (pts[45][0]+side_len,pts[45][1] + half_len)
    areas.append((area_5_pt1,area_5_pt2))

    area_10_pt1 = (pts[8][0]-half_len,pts[8][1] - side_len)
    area_10_pt2 = (pts[8][0] + half_len,pts[8][1])
    areas.append((area_10_pt1,area_10_pt2))

    return areas

def getFlowFeatures(flow,areas):
    sum_features=[]
    
    for area in areas:
        x1,y1 = int(area[0][0]),int(area[0][1])
        x2,y2 = int(area[1][0]),int(area[1][1])
        parition = flow[y1:y2,x1:x2]
        temp = np.reshape(parition,(-1,2))
        deltaX= temp[:,0]
        deltaY = temp[:,1]
        areaFlowXSum= deltaX.sum()
        areaFlowYSum = deltaY.sum()
        sum_features.append([areaFlowXSum,areaFlowYSum])
        midx,midy = getmid_pt((x1,y1),(x2,y2))
        lines.append([[midx,midy],[int(midx+areaFlowXSum),int(midy+areaFlowYSum)]])
    sum_features = np.array(sum_features)
    return sum_features.flatten()

if __name__ == "__main__":

    dir = "EP03_2"
    step = 4
    height,width,_ = cv2.imread(dir+"\\"+dir+"_"+str(1)+".jpg").shape
    sum_flow = np.zeros((height,width,2))
    MobileNet_detector = face_detector.landmark_detector("MobileNet")

    '''for i in range(2, len(os.listdir(dir))):
        prev = cv2.imread(dir+"\\"+dir+"_"+str(i-1)+".jpg")
        next = cv2.imread(dir+"\\"+dir+"_"+str(i)+".jpg")
        print(dir+"\\"+dir+"_"+str(i)+".jpg")
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
        optical_flow= cv2.optflow.DualTVL1OpticalFlow_create(nscales=1,epsilon=0.05,warps=1)
        flow = optical_flow.calc(prev_gray, next_gray, None)
        landmarks = MobileNet_detector.detect_landmarks("Retinalface",dir+"\\"+dir+"_"+str(i)+".jpg")
        tenAreas = facial_area_coordinate(landmarks)
        getFlowFeatures(flow,tenAreas)
        for area in tenAreas:
            x1,y1 = area[0][0],area[0][1]
            x2,y2 = area[1][0],area[1][1]
            cv2.rectangle(prev,(int(x1),int(y1)),(int(x2),int(y2)),color=(0,255,0),thickness=1)

        for line in lines:
            cv2.line(prev,line[0],line[1],color=(0,255,0),thickness= 1)
        lines = []'''
    
    for i in range(2,5):
        prev = cv2.imread(dir+"\\"+dir+"_"+str(i-1)+".jpg")
        next = cv2.imread(dir+"\\"+dir+"_"+str(i)+".jpg")
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
        optical_flow= cv2.optflow.DualTVL1OpticalFlow_create(nscales=1,epsilon=0.05,warps=1)
        flow = optical_flow.calc(prev_gray, next_gray, None)
        temp_flow_img = draw_flow(prev_gray,flow)
        landmarks = MobileNet_detector.detect_landmarks("Retinalface",dir+"\\"+dir+"_"+str(i)+".jpg")
        bbox = MobileNet_detector.getfacebb(dir+"\\"+dir+"_"+str(i)+".jpg")
        print(bbox[0][0])
        print(bbox[0])
        x1,y1 = int(bbox[0][0][0]),int(bbox[0][0][1])
        x2,y2 = int(bbox[0][1][0]),int(bbox[0][1][1])
        cv2.rectangle(next,(x1,y1),(x2,y2),(0,0,255),thickness=3)
        for landmark in landmarks:
            x1,y1 = int(landmark[0]),int(landmark[1])
            cv2.circle(next,(x1,y1),3,(0,255,0),thickness=3)
        cv2.imshow("temp",next)
        cv2.waitKey(0)
        tenAreas = facial_area_coordinate(landmarks)
        for area in tenAreas:
            x1,y1 = int(area[0][0]),int(area[0][1])
            x2,y2 = int(area[1][0]),int(area[1][1])
            cv2.rectangle(next,(x1,y1),(x2,y2),color=(0,0,255),thickness=2)
        feature = getFlowFeatures(flow,tenAreas)
        for line in lines:
            cv2.line(next,line[0],line[1],color=(0,255,0),thickness=1)
        cv2.imshow("next",next)
        cv2.waitKey(0)
        lines = []
        temp_flow_img = flow_vis.flow_to_color(flow)
        sum_flow = sum_flow + flow
    
    origin = cv2.imread(dir+"\\"+dir+"_"+str(62)+".jpg")
    mask = cv2.cvtColor(origin,cv2.COLOR_BGR2GRAY)

    landmarks = MobileNet_detector.detect_landmarks("Retinalface",dir+"\\"+dir+"_"+str(62)+".jpg")
    tenAreas = facial_area_coordinate(landmarks)
    
    getFlowFeatures(flow,tenAreas)
    flowImage = draw_flow(mask,sum_flow)
    for area in tenAreas:
        x1,y1 = int(area[0][0]),int(area[0][1])
        x2,y2 = int(area[1][0]),int(area[1][1])
        cv2.rectangle(origin,(x1,y1),(x2,y2),color=(0,0,255),thickness=2)
    for line in lines:
        print(line[1])
        print(line[0])
        cv2.line(origin,line[0],line[1],color=(0,255,0),thickness= 1)
    cv2.imshow("mask",origin)
    cv2.waitKey(0)
    cv2.imshow("color",flowImage)
    cv2.waitKey(0)
    flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
    cv2.imshow("color",flow_color)
    cv2.waitKey(0)
    facial_flow = sum_flow[y1:y2,x1:x2]

    
               
