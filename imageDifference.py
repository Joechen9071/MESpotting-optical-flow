import face_detector
import numpy as np
import cv2
import os 


side_len = 0
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

def image_diff(img1,img2,img3):
    img_t = cv2.imread(img1)
    img_lambda = cv2.imread(img2)
    img_eplison = cv2.imread(img3)

    img_t = cv2.cvtColor(img_t,cv2.COLOR_BGR2GRAY)
    height,width = img_t.shape
    img_lambda = cv2.cvtColor(img_lambda,cv2.COLOR_BGR2GRAY)
    img_eplison = cv2.cvtColor(img_eplison,cv2.COLOR_BGR2GRAY)
    ones_mat = np.ones((height,width))
    numerator = np.abs(img_t - img_lambda) + ones_mat
    denomator = np.abs(img_t - img_eplison) + ones_mat

    ''' 
    numerator = np.abs(img_t - img_lambda)
    denomator = np.abs((img_t - img_eplison))

    numerator = numerator + 1
    denomator = denomator + 1 '''
    print(numerator/denomator)
    return numerator/denomator

def getimgdiffFeatures(imgdiff,areas):
    avg_features=[]
    
    for area in areas:
        x1,y1 = int(area[0][0]),int(area[0][1])
        x2,y2 = int(area[1][0]),int(area[1][1])
        parition = imgdiff[y1:y2,x1:x2]
        temp = np.mean(parition)
        avg_features.append(temp)
    
    avg_features = np.array(avg_features)

    return list(avg_features.flatten())
if __name__ == "__main__":
    dir = "EP03_2"
    detector = face_detector.landmark_detector("MobileNet")
    for i in range(5,len(os.listdir(dir))):
        imgt = dir+"\\"+dir+"_"+str(i)+".jpg"
        img_lam = dir+"\\"+dir+"_"+str(i-4)+".jpg"
        img_epl = dir+"\\"+dir+"_"+str(i-2)+".jpg"
        landmarks = detector.detect_landmarks("MTCNN",imgt)
        diff = image_diff(imgt,img_lam,img_epl)
        cv2.imshow("diff",diff)
        cv2.waitKey(0)
        areas = facial_area_coordinate(landmarks)
        feature = getimgdiffFeatures(diff,areas)
        print(np.array(feature).shape)
    print(image_diff("EP03_2\\EP03_2_5.jpg","EP03_2\\EP03_2_1.jpg","EP03_2\\EP03_2_3.jpg").shape)