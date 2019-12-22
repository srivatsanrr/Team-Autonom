# coding: utf-8
# # Object Detection Demo
# License: Apache License 2.0 (https://github.com/tensorflow/models/blob/master/LICENSE)
# source: https://github.com/tensorflow/models
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import time
import smtplib
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
#configure smptblib

'''s = smtplib.SMTP('smtp.gmail.com', 587) 

s.starttls()
s.login("arcvitvellore@gamil.com", "ARC@VIT2019")
message = "Road accident at 'LOCATION'"
s.sendmail("arcvitvellore@gamil.com","krshnakle@gmail.com",message)
s.quit()'''

# from grabscreen import grab_screen
import cv2

import pyttsx3
import sys

# Instantiate OCV kalman filter
class KalmanFilter:

    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def Estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted

engine = pyttsx3.init() # object creation
rate = engine.getProperty('rate')   # getting details of current speaking rate
                                        #printing current voice rate
engine.setProperty('rate', 200)     # setting up new voice rate



volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
                                         #printing current volume level
engine.setProperty('volume',1.5)    # setting up volume level  between 0 and 1


voices = engine.getProperty('voices')       #getting details of current voice
#engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
engine.setProperty('voice', voices[1].id)

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

from utils import label_map_util
from utils import visualization_utils as vis_util


# # Model preparation 
# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

collect_path = []
pred_path = []

def remap( x, oMin, oMax, nMin, nMax ):

    #range check
    

    #check reversed input range
    reverseInput = False
    oldMin = min( oMin, oMax )
    oldMax = max( oMin, oMax )
    if not oldMin == oMin:
        reverseInput = True

    #check reversed output range
    reverseOutput = False   
    newMin = min( nMin, nMax )
    newMax = max( nMin, nMax )
    if not newMin == nMin :
        reverseOutput = True

    portion = (x-oldMin)*(newMax-newMin)/(oldMax-oldMin)
    if reverseInput:
        portion = (oldMax-x)*(newMax-newMin)/(oldMax-oldMin)

    result = portion + newMin
    if reverseOutput:
        result = newMax - portion

    return result

# ## Download Model
'''opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())'''


# ## Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
# _,fr=cap.read()
#       prev=fr

  
with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
     # Create Kalman Filter Object
    kfObj = KalmanFilter()
    predictedCoords = np.zeros((2, 1), np.float32)
    cap=cv2.VideoCapture('testdata1.mp4') #.mp4
    fps = cap.get(cv2.CAP_PROP_FPS)
     
    timestamps = [cap.get(cv2.CAP_PROP_POS_MSEC)]
    calc_timestamps = [0.0]
    print(fps)  
    pmidy=0
    cntr = 0
    cntl = 0
    l = []
    while True:
      
      _,fr=cap.read()
      
      #screen = cv2.resize(grab_screen(region=(0,40,1280,745)), (WIDTH,HEIGHT))
      timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC))
      calc_timestamps.append(calc_timestamps[-1] + 1000/fps)
     
      screen = cv2.resize(fr, (800,450))
      image_np = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
      image_np_expanded = np.expand_dims(image_np, axis=0)
      image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
      # Each box represents a part of the image where a particular object was detected.
      boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
      # Each score represent how level of confidence for each of the objects.
      # Score is shown on the result image, together with the class label.
      scores = detection_graph.get_tensor_by_name('detection_scores:0')
      classes = detection_graph.get_tensor_by_name('detection_classes:0')
      num_detections = detection_graph.get_tensor_by_name('num_detections:0')
      # Actual detection.
      (boxes, scores, classes, num_detections) = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})
      # Visualization of the results of a detection.
      vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)

    
      for i,b in enumerate(boxes[0]):
        #                 car                    bus                  truck
        if classes[0][i] == 3 or classes[0][i] == 6 or classes[0][i] == 8 or classes[0][i]==1:
          if scores[0][i] >= 0.5:
            mid_x = (boxes[0][i][1]+boxes[0][i][3])/2
            mid_y = (boxes[0][i][0]+boxes[0][i][2])/2
            predictedCoords = kfObj.Estimate(mid_x, mid_y)
            

           
            
            pred_x = remap(predictedCoords[0],0,1.0,0,800)
            pred_y = remap(predictedCoords[1],0,1.0,0,450)
            ac_x = remap(mid_x,0,1.0,0,800)
            ac_y = remap(mid_y,0,1.0,0,450)
            collect_path.append((int(ac_x),int(ac_y)))
            pred_path.append((int(pred_x),int(pred_y)))
            print("Actual Centroid:\n")
            print(int(ac_x),int(ac_y))
            print("Predicted Centroid")
            print(int(pred_x),int(pred_y))
            
            cv2.line(image_np,(int(ac_x),int(ac_y)),(int(pred_x),int(pred_y)),(0,0,255),9)
                
           
            


           # print(mid_x)
            #print(mid_y)
            mag = pmidy - mid_y
            pmidy = mid_y
            mag*=1000
            pmidyi = int(mag)
            l.append(pmidyi)

            if len(l)>20:
        
              for i in range(0,len(l)):
                if(l[i]<=0):
                  cntl+=1
                else:
                  cntr+=1
              

              print(cntl,cntr) 
              if cntl>11 and cntl>cntr:
          
                """ RATE"""
                   #changing index, changes voices. 1 for female
                #print("turn right")
                
                #engine.say("Turn Left")
#engine.say('My current speaking rate is ' + str(rate))
                engine.runAndWait()
                engine.stop()
              elif cntr>11 and cntr>cntl:
                '''print("turn left")
              else:
                '''
                #print("straight")
              
              cntr=0
              cntl=0
              l = []

            #image_np = cv2.line(image_np, (int(mid_y*100),int(mid_x*100)), (int(pmidyi),int(mid_x*100)),(0, 255, 0), 9)

            apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1]))**4),1)
            cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x*800),int(mid_y*450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            
            #cv2.putText(image_np, '{}',format(apx_distance))
            if apx_distance <=0.5:
              for jimjam in range(0,len(calc_timestamps)):
                if calc_timestamps[jimjam] == 0:
                  continue  
                else:

                  speed = float(apx_distance)/float(abs(calc_timestamps[jimjam]-calc_timestamps[jimjam-1]))
                  print("Speed",speed*10000)
              if len(pred_path)>10: 
                '''for jimjam in range(0,len(pred_path)):
                  cv2.line(image_np,collect_path[jimjam],(pred_path[jimjam][0]+50,pred_path[jimjam][1]+50),(0,0,255),9)
                '''
                collect_path = []
                pred_path = []
              

              if mid_x > 0.3 and mid_x < 0.7:
                cv2.putText(image_np, 'WARNING!!!', (400,225), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (255,0,255), 9)
                print("ESERVICE ALERTED CRASH AT ______")
                
            
      
      cv2.imshow('window',cv2.resize(image_np,(800,450)))
      if cv2.waitKey(25) & 0xFF == ord('q'):
          cv2.destroyAllWindows()
          break


