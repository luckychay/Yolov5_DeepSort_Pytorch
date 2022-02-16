'''
Description: 
Version: 
Author: Xuanying Chen
Date: 2022-02-16 17:09:59
LastEditTime: 2022-02-16 17:14:40
'''
import cv2
import numpy as np
import rospy
from cm_transport.msg import CustomCImage
from yolov5.utils.augmentations import letterbox

class LoadRosTopic:  # for inference
    # YOLOv5 rostopic dataloader, i.e. `python detect.py --source 1`
    def __init__(self, topic='/usb_cam/compressed', img_size=640, stride=32, auto=True):
        self.img_size = img_size
        self.stride = stride
        self.auto = auto
        self.topic = topic
        self.image = None

        rospy.init_node('listener', anonymous=True)
        rospy.Subscriber(self.topic, CustomCImage, self.callback, queue_size = 3)

    def callback(self,data):
        np_arr = np.fromstring(data.image.data, np.uint8)
        self.image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.tram_status = data.tram_status.status
        rospy.loginfo(rospy.get_caller_id() + "I heard %d",data.tram_status.status)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        
        self.count += 1
        # Read frame
        img0 = self.image.copy()
        
        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride,auto=self.auto)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return self.topic, img, img0, None, '', self.tram_status      

    def __len__(self):
        return 0