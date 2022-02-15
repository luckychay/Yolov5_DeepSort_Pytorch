# limit the number of cpus used by high performance libraries
import os

from cv2 import WINDOW_NORMAL
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from PIL import ImageEnhance
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def detect(opt):
    out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok, roi, do_entrance_counting, do_process= \
        opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok, opt.roi, opt.en_counting,opt.process
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    device = select_device(opt.device)
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(deep_sort_model,
                        device,
                        max_dist=cfg.DEEPSORT.MAX_DIST,
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        )

    # Initialize
    
    half &= device.type != 'cpu'  # half precision only supported on CUDA

    # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # its own .txt file. Hence, in that case, the output folder is not restored
    if not evaluate:
        if os.path.exists(out):
            pass
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

    # Prepare for entrance counting
    if do_entrance_counting:
        shape = dataset.imgs[0].shape
        in_id_list = list()
        out_id_list = list()
        in_flag = dict()
        out_flag = dict()
        prev_center = dict()
        count_str = ""
        # Determination of the lines may be tricky
        entrance1 =  tuple(map(int,[0, shape[0] / 2.0, shape[1], shape[0] / 2.0]))
        entrance2 =  tuple(map(int,[0, shape[0] / 1.8, shape[1], shape[0] / 1.8]))

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        
        if do_process:
        # do equilized histgram for BCHW RGB images
            for i in range(img.shape[0]): 
                pic = img[i][::-1,...].transpose((1,2,0)) # move the channel dim to the last and convert into BGR(Opencv needs)
                # (b,g,r) = cv2.split(pic)
                # b = cv2.GaussianBlur(b,(7,7),0)
                # g = cv2.GaussianBlur(g,(7,7),0)
                # r = cv2.GaussianBlur(r,(7,7),0)

                # bH = cv2.equalizeHist(b)
                # gH = cv2.equalizeHist(g)
                # rH = cv2.equalizeHist(r)
                # pic = cv2.merge((bH,gH,rH))
                # cv2.imshow("equilized",pic)
                # cv2.waitKey(1)
                # img[i] = pic[...,::-1].transpose(2,0,1)

                lab= cv2.cvtColor(pic, cv2.COLOR_BGR2LAB)

                #-----Splitting the LAB image to different channels-------------------------
                l, a, b = cv2.split(lab)

                #-----Applying CLAHE to L-channel-------------------------------------------
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                cl = clahe.apply(l)

                #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
                limg = cv2.merge((cl,a,b))

                #-----Converting image from LAB Color model to RGB model--------------------
                final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

                # cv2.imshow('final', final)
                # cv2.waitKey(1)
                img[i] = final[...,::-1].transpose(2,0,1)
                im0s[i] = final.copy()
        # Use roi to filter 
        if roi:
            origin = (50,180)
            img[...,0:origin[1],:] = 0
            img[...,origin[1]::,0:origin[0]] = 0

        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                s += f'webcam{i}: '
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            s += '%gx%g ' % img.shape[2:]  # print string

            annotator = Annotator(im0, line_width=2, pil=not ascii)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
  
                # pass detections to deepsort
                t4 = time_sync()
                outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4
                # draw boxes for visualization
                if len(outputs) > 0:

                    for j, (output, conf) in enumerate(zip(outputs, confs)):

                        bboxes = output[0:4]
                        track_id = output[4]
                        cls = output[5]

                        c = int(cls)  # integer class
                        label = f'{track_id} {names[c]} {conf:.2f}'
                        annotator.box_label(bboxes, label, color=colors(c, True))

                        # Use two line to do entrance counting
                        if do_entrance_counting and cls in opt.classes:
                            entrance_y1 = entrance1[1] 
                            entrance_y2 = entrance2[1]

                            if track_id < 0: continue

                            x1, y1, x2,y2 = bboxes
                            center_x = (x1 + x2)/2.
                            center_y = (y1 + y2)/2.

                            if track_id in prev_center:

                                # In number counting 
                                if prev_center[track_id][1] <= entrance_y1 and \
                                center_y > entrance_y1:
                                    in_flag[track_id] = 1
                                elif prev_center[track_id][1] <= entrance_y2 and \
                                center_y > entrance_y2 and in_flag[track_id] == 1:
                                    in_id_list.append(track_id)
                                    in_flag[track_id] = 0

                                # Out number counting
                                elif prev_center[track_id][1] >= entrance_y2 and \
                                center_y < entrance_y2:
                                    out_flag[track_id] = 1
                                elif prev_center[track_id][1] >= entrance_y1 and \
                                center_y < entrance_y1 and out_flag[track_id] == 1:
                                    out_id_list.append(track_id)
                                    out_flag[track_id] = 0

                                prev_center[track_id] = [center_x, center_y]
                            else:
                                prev_center[track_id] = [center_x, center_y]
                                in_flag[track_id] = 0
                                out_flag[track_id] = 0
                            
                            count_str = f"In: {len(in_id_list)}, Out: {len(out_id_list)}"
                            print(count_str)

                        if save_txt:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            # Write MOT compliant results to file
                            with open(txt_path, 'a') as f:
                                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                               bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))

                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

            else:
                deepsort.increment_ages()
                # LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
            if show_vid:
                
                lw = 3
                tf = max(lw - 1, 1)
                w, h = cv2.getTextSize(s, 0, fontScale=lw / 3, thickness=tf)[0] 
                p1 = (0,0)
                p2 = (p1[0] + int(w), p1[1]+int(h)+10)
                cv2.rectangle(im0, p1, p2, (0,240,240), -1, cv2.LINE_AA)  # filled
                cv2.putText(im0, s, (p1[0], p1[1]+h+3), 0, lw / 3, (255,255,255),
                            thickness=tf, lineType=cv2.LINE_AA)

                if do_entrance_counting:
                    w, h = cv2.getTextSize(count_str, 0, fontScale=lw / 3, thickness=tf)[0]
                    p1 = (0,p2[1])
                    p2 = (p1[0] + int(w), p1[1]+int(h)+10)
                    cv2.rectangle(im0, p1, p2, (240,240,0), -1, cv2.LINE_AA)  # filled
                    cv2.putText(im0, count_str, (p1[0], p1[1]+h+3), 0, lw / 3, (255,255,255),
                                thickness=tf, lineType=cv2.LINE_AA)
                    cv2.rectangle(im0,entrance1[0:2],entrance1[2:4],(0,255,255),1)
                    cv2.rectangle(im0,entrance2[0:2],entrance2[2:4],(0,255,255),1)

                # cv2.namedWindow(str(p),WINDOW_NORMAL)  
                # cv2.resizeWindow(str(p),640,480)  
                cv2.imshow(str(p), im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_vid:
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path += '.mp4'

                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

                vid_writer.write(im0)


    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
        per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        print('Results saved to %s' % save_path)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--evaluate', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'inference', help='save results to project/name')
    parser.add_argument('--name', default='output', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--roi', action='store_true', help='turn on roi filter')
    parser.add_argument('--en_counting', action='store_true', help='turn on entrance counting')
    parser.add_argument('--process', action='store_true', help='turn on image processing')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
