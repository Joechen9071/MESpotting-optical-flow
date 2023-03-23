import face_detector
import cv2
import numpy as np
import sklearn.svm as svm
import joblib
side_len = 0
def find_cluster(arrray,elem):
    found_elem = False
    intervals = []
    current_idx = 0
    for i in range(len(arrray)):

        if arrray[i] == elem and not found_elem:
            found_elem = True
            current_idx = i
        elif found_elem and arrray[i] != elem:
            intervals.append((current_idx,i))
            found_elem = False
    if found_elem:
        intervals.append((current_idx,len(arrray)))
    return intervals

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
    sum_features = np.array(sum_features)
    return sum_features.flatten()

def readframes(video_path):
    cap = cv2.VideoCapture(video_path)
    count = 1
    read = True
    while read:
        ret,frame = cap.read()
        if not ret:
            break
        else:
            cv2.imwrite("input\\input_"+str(count)+".jpg",frame)
            count += 1
    return count

def compute_optical_flow(dir,count):
    opti_flow = cv2.optflow.DualTVL1OpticalFlow_create(nscales=1,epsilon=0.05,warps=1)
    landmark_detector = face_detector.landmark_detector('MobileNet')
    features = []
    for i in range(2,count):
        prev = cv2.imread(dir+"\\input_"+str(i-1)+".jpg")
        next = cv2.imread(dir+"\\input_"+str(i)+".jpg")
        prev_gray = cv2.cvtColor(prev,cv2.COLOR_BGR2GRAY)
        next_gray = cv2.cvtColor(next,cv2.COLOR_BGR2GRAY)
        flow = opti_flow.calc(prev_gray,next_gray,None)
        landmarks = landmark_detector.detect_landmarks("Retinalface",dir+"\\input_"+str(i)+".jpg")
        for lm in landmarks:
            x,y= int(lm[0]),int(lm[1])
            cv2.circle(next,(x,y),1,(0,0,255),1)
        cv2.imshow("img",next)
        cv2.waitKey(1)
        tenArea = facial_area_coordinate(landmarks)
        feature = getFlowFeatures(flow,tenArea)
        features.append(feature)
    features = np.array(features)
    return features

if __name__ == "__main__":
    video_count = readframes("EP01_5.avi")
    features = compute_optical_flow("input",video_count)
    clf = joblib.load("optical_flow_svm.joblib")

    predict_result = clf.predict(features)
    cluster = find_cluster(predict_result,1)
    print(cluster)