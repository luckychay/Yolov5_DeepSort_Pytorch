 ###
 # @Description: --
 # @Version: 
 # @Author: Xuanying Chen
 # @Date: 2022-02-03 11:00:53
 # @LastEditTime: 2022-02-07 23:32:07
### 
#!/bin/bash
# if use compressed image, add "\&type=ros_compressed"  
# --yolo_model yolov5/weights/crowdhuman_yolov5m.pt

python3 track.py --source http://localhost:8080/stream?topic=/usb_cam/image_raw \
--save-vid --classes 0 --en_counting --exist-ok --yolo_model yolov5/weights/crowdhuman_yolov5m.pt --process
