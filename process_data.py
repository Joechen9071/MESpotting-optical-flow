import sklearn
import numpy as np
import processCSV
import os
import sklearn.svm as svm
import joblib

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

if __name__ =="__main__":
    if not os.path.isfile("optical_flow_svm.joblib"):
        root_dir = "optical_data"
        data = []
        single_video = []
        for dataset in os.listdir(root_dir):
            for sub in os.listdir(root_dir+"\\"+dataset):
                for file in os.listdir(root_dir+"\\"+dataset+"\\"+sub):
                    data = data + processCSV.get_vector_csv(root_dir+"\\"+dataset+"\\"+sub+"\\"+file)
                    if len(single_video) == 0:
                        single_video = processCSV.get_vector_csv_without_label(root_dir+"\\"+dataset+"\\"+sub+"\\"+file)
        single_video = processCSV.get_vector_csv_without_label(r"optical_data\casme1_sectionA\sub01\EP01_5_single.csv")
        data = np.array(data)
        ME_frame = data[np.in1d(data[:, 20], [1])]
        non_ME_frame = data[np.in1d(data[:, 20], [-1])]
        np.random.shuffle(non_ME_frame)
        non_ME_frame =non_ME_frame[:len(ME_frame)]
        data_set = np.concatenate((ME_frame,non_ME_frame))
        np.random.shuffle(data_set)
        label = data_set[:, -1]
        feature = data_set[:,:-1]
        clf = svm.SVC(gamma='auto')
        clf.fit(feature,label)

        joblib.dump(clf,"optical_flow_svm.joblib")
    else:
        print("Found fitted model")
        clf = joblib.load("optical_flow_svm.joblib")


    truth  = processCSV.getGroundTruth("casme1_sectionA.csv",4)
    temp = {}
    for video in truth:
        temp[video[0]] = video[1]
    detection_result = {}
    root_dataset = "optical_data\\casme1_sectionA"
    for sub in os.listdir(root_dataset):
        for datacsv in os.listdir(root_dataset + "\\" + sub):
            video_name= datacsv.split("_single")[0]
            single_video = processCSV.get_vector_csv_without_label(root_dataset+"\\"+sub+"\\"+datacsv)
            prediction = clf.predict(single_video)
            found_cluster = find_cluster(prediction,1)
            detection_result[sub + "-"+ video_name] = found_cluster
    
    result_keys = list(detection_result.keys())
    accuracy = 0.0
    count = 0
    for key in result_keys:
        for interval_idx in range(0,len(temp[key])):
            
            list_truth_frames = list(range(temp[key][interval_idx][0],temp[key][interval_idx][1] + 1))
            list_detected_frame = list(range(detection_result[key][interval_idx][0],detection_result[key][interval_idx][1] + 1))
            accuracy += len(set(list_truth_frames)&set(list_detected_frame))/len(set(list_truth_frames)|set(list_detected_frame)) 
            count += 1
    accuracy = accuracy/ count
    print("CASME1 section A: "+str(accuracy))
    '''    
    for k in detection_result:
        print(k + " " + str(detection_result[k]))'''

    truth  = processCSV.getGroundTruth("casme1_sectionB.csv",4)
    for video in truth:
        temp[video[0]] = video[1]

    detection_result = {}
    root_dataset = "optical_data\\casme1_sectionB"
    for sub in os.listdir(root_dataset):
        for datacsv in os.listdir(root_dataset + "\\" + sub):
            video_name= datacsv.split("_single")[0]
            single_video = processCSV.get_vector_csv_without_label(root_dataset+"\\"+sub+"\\"+datacsv)
            prediction = clf.predict(single_video)
            found_cluster = find_cluster(prediction,1)
            detection_result[sub + "-"+ video_name] = found_cluster
    
    result_keys = list(detection_result.keys())
    accuracy = 0.0
    count = 0
    for key in result_keys:
        for interval_idx in range(0,len(temp[key])):
            
            list_truth_frames = list(range(temp[key][interval_idx][0],temp[key][interval_idx][1] + 1))
            list_detected_frame = list(range(detection_result[key][interval_idx][0],detection_result[key][interval_idx][1] + 1))
            accuracy += len(set(list_truth_frames)&set(list_detected_frame))/len(set(list_truth_frames)|set(list_detected_frame)) 
            count += 1
    accuracy = accuracy/ count
    
    print("CASME1 section B: "+str(accuracy))


    '''for k in detection_result:
        print(k + " "+str(detection_result[k]))'''


    truth  = processCSV.getGroundTruth("casme2_raw.csv",4)
    for video in truth:
        temp[video[0]] = video[1]

    root_dataset = "optical_data\\casme_raw"
    for sub in os.listdir(root_dataset):
        for datacsv in os.listdir(root_dataset + "\\" + sub):
            video_name= datacsv.split("_single")[0]
            single_video = processCSV.get_vector_csv_without_label(root_dataset+"\\"+sub+"\\"+datacsv)
            prediction = clf.predict(single_video)
            found_cluster = find_cluster(prediction,1)
            detection_result[sub + "-"+ video_name] = found_cluster

    '''for k in detection_result:
        print(k + " "+str(detection_result[k]))'''

    result_keys = list(detection_result.keys())
    accuracy = 0.0
    count = 0
    for key in result_keys:
        for interval_idx in range(0,len(temp[key])):
            
            list_truth_frames = list(range(temp[key][interval_idx][0],temp[key][interval_idx][1] + 1))
            list_detected_frame = list(range(detection_result[key][interval_idx][0],detection_result[key][interval_idx][1] + 1))
            accuracy += len(set(list_truth_frames)&set(list_detected_frame))/len(set(list_truth_frames)|set(list_detected_frame)) 
            count += 1
    accuracy = accuracy/ count
    
    print("CASME2: "+str(accuracy))
