import csv
import os
import matplotlib as plt

'''
This function assume dataset to following structure
Subject, onset_frame, Apex_frame 
3 columns based file.
'''

def read_CASME_csv(file_path):
    res = []
    with open(file_path) as csv_file:
        csv_reader = csv.reader(csv_file,delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count = line_count + 1
            else:
                lambda_value = (int(row[3]) - int(row[2]) + 1) * 2
                row.append(lambda_value)
                row[0] = f"{int(row[0]):02}"
                row[3] = int(row[3])
                res.append(row)
    return res

def append_to_row(file_path,array):
    if not os.path.isfile(file_path):
        with open(file_path,'w',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(array)
            f.close()
    else:  
        with open(file_path,'a',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(array)
            f.close()
def get_vector_csv_with_label(filename,label):
    data = []
    count = 0
    with open(filename,'r') as f:
        reader = csv.reader(f)
        for row in reader:
            converted_row = list(map(float,row))
            converted_row.append(label)
            data.append(converted_row)
            count += 1
    print(str(label) + " has " + str(count) + " samples")
    return data
    
def get_vector_csv(filename):
    data = []
    count = 0
    with open(filename,'r') as f:
        reader = csv.reader(f)
        for row in reader:
            converted_row = list(map(float,row))
            data.append(converted_row)
            count += 1
    return data
def get_vector_csv_without_label(filename):
    data = []
    count = 0
    with open(filename,'r') as f:
        reader = csv.reader(f)
        for row in reader:
            converted_row = list(map(float,row))
            data.append(converted_row[0:20])
            count += 1
    return data

def load_csv(filename):
    data = []
    with open(filename) as csv_file:
        reader = csv.reader(csv_file,delimiter=',')
        for i in reader:
            data.append(i)
    return data
def getKeys(li):
    keys = []
    for i in li:
        keys.append(list(i.keys())[0])
    return keys
def getKeyidx(li,key):
    for i in range(0,len(li)):
        if list(li[i].keys())[0] == key:
            return i
        else:
            continue
def getDictfromcsv(filename,lambda_val):
    dir_dict = {}
    res = read_CASME_csv(filename)
    for row in res:
        if row[0] not in dir_dict:
            dir_dict[row[0]] = []
        if row[1] in getKeys(dir_dict[row[0]]):
            idx = getKeyidx(dir_dict[row[0]],row[1])
            dir_dict[row[0]][idx][row[1]].append((row[3] - lambda_val, row[3]+lambda_val)) 
        else:
            dir_dict[row[0]].append({row[1]: [(row[3] - lambda_val, row[3]+lambda_val)]})
    return dir_dict

def getGroundTruth(filename,lambda_val):
    file_dir = getDictfromcsv(filename,lambda_val)
    result = []
    for key in list(file_dir.keys()):
        for videos in file_dir[key]:
            for video in videos:
                partial_res = []
                list_of_list = [list(elem) for elem in videos[video]]
                partial_res.append("sub" + key+"-"+video)
                partial_res.append(list_of_list)
                result.append(partial_res)
    return result

def changelabel(vectors,new_label):
    new_label_data = []
    for vector in vectors:
        temp = vector[0:20]
        temp.append(new_label)
        new_label_data.append(temp)
    return new_label_data
if __name__ == "__main__":
   data = get_vector_csv("processed_data\\normalized_non_ME_data.csv")
   data = changelabel(data,-1)
   print(data)
   

       
   
   