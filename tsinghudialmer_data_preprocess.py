import cv2
import os
import json

annotation_data, meta_data, intermidiate_data = [],[],[]

#"/home/ubuntu/tdcb_leftImg8bit_train/leftImg8bit/train/tdcb_labelData_train/labelData/train/tsinghuaDaimlerDataset/"
def load_labels(json_folder_path="/home/ubuntu/tdcb_leftImg8bit_train/leftImg8bit/train/tdcb_labelData_train/labelData/train/tsinghuaDaimlerDataset/"):
    json_files = [ x for x in os.listdir(json_folder_path) if x.endswith("json") ]
    json_data = list()
    for json_file in json_files:
        json_file_path = os.path.join(json_folder_path, json_file)
        with open (json_file_path, "r") as f:
            json_data.append(json.load(f))
    for element in json_data:
        intermidiate_data=[]
        for data in element['children']:
            intermidiate_data.append([data['minrow'], data['mincol'], data['maxrow'], data['maxcol']])
        meta_data.append(intermidiate_data)
    annotation_data.append(meta_data)
    return annotation_data[0]
    
    
def load_images_from_folder (folder='/home/ubuntu/tdcb_leftImg8bit_train/leftImg8bit/train/tsinghuaDaimlerDataset/'):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

'''print(load_labels())
print(len(load_labels()))'''


    
