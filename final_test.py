import cv2
import os
import json
import numpy as np


final_list = []

#"/home/ubuntu/tdcb_leftImg8bit_train/leftImg8bit/train/tdcb_labelData_train/labelData/train/tsinghuaDaimlerDataset/"
def load_labels(filename, json_folder_path="/home/ubuntu/tdcb_leftImg8bit_train/leftImg8bit/train/tdcb_labelData_train/labelData/train/tsinghuaDaimlerDataset/"):
    annotation_data, meta_data, intermidiate_data = [],[],[]
    json_files = [ x for x in os.listdir(json_folder_path) if x == '{}'.format(filename) ]
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
    i = 0
    for filename in os.listdir(folder):
        if filename.endswith('png') :
            img = cv2.imread(os.path.join(folder,filename))
            filename_to_be_passed = filename[:-15] + 'labelData.json'
            labels = load_labels(filename_to_be_passed)
            images.append([img, labels[0]])
            print(i)
            i = i+1
            print("<<<<<<<<<<<------------------>>>>>>>>>>>>>>>")
    return images[0]


training_data = np.save('training_data.npy', load_images_from_folder(), allow_pickle=True)

#print(load_images_from_folder())
print(len(load_images_from_folder()))

#print(load_images_from_folder()[1][1])
#print(load_images_from_folder()[1][0].shape) #####IMAGE
#print(load_images_from_folder()[0][1]) #####LABEL

image = cv2.resize(load_images_from_folder()[1][0], (640, 480))
label = load_images_from_folder()[0][1]

print(label)

cv2.rectangle(image, (label[0][0], label[0][1]), (label[0][2], label[0][3]), (255,0,0), 5)

cv2.imshow('frame1', image)
cv2.waitKey(0)

