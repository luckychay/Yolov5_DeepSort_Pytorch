 ###
 # @Description: --
 # @Version: 
 # @Author: Xuanying Chen
 # @Date: 2022-02-03 11:00:53
 # @LastEditTime: 2022-02-06 13:41:20
### 
#!/bin/bash
#\&type=ros_compressed  
python3 track.py --source http://localhost:8080/stream?topic=/usb_cam/image_raw\&type=ros_compressed --yolo_model yolov5/weights/crowdhuman_yolov5m.pt --classes 0 --en_counting
