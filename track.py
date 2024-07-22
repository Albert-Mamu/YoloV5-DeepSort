import sys

sys.path.insert(0, './yolov5')

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
countFace = 0

def bbox_rel(*xyxy):
    """" Calculate boundary values """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Add different color borders
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    """
    Draw detection box
    """
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # Object ID
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

    return img

def detect_faces(face_cascade, img, x0, y0, w0, h0, save_path):
    """
    Detect face on person
    """    
    global countFace

    imgCrop = None
    imgCrop = img[y0:y0+h0, x0:x0+w0]

    gray = cv2.cvtColor(imgCrop, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 

    if len(faces) >= 1:
        for (x,y,w,h) in faces: 
            countFace=countFace+1
            # save faces
            imgSave = imgCrop[y:y+h, x:x+w]
            save_img = save_path[:-3]
            save_img += str(countFace) + ".jpg"
            cv2.imwrite(save_img, imgSave)

    return img

def detect(opt, save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # Load Face
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # initialization deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # deepsort = DeepSort(opt.weights_deepsort)

    # Initialize the device
    device = select_device(opt.device)
    
    # Clear device memory
    torch.cuda.set_per_process_memory_fraction(0.5, 0)
    torch.cuda.empty_cache()

    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Loading the model
    model = torch.load(weights, map_location=device)[
        'model'].float()
    model.to(device).eval()
    if half:
        model.half()

    # Reading in data
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get label name and color
    names = model.module.names if hasattr(model, 'module') else model.names

    # run
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # Initialize the image
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # predict
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # application NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Run the test
        for i, det in enumerate(pred):  # Frame by frame detection
            if webcam:
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Redefine the border
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                startFace = False
                facesArray = [0] * 100

                # Output
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # Traverse each category

                    if names[int(c)] == 'person':   # Person detection
                        startFace = True;
                        facesArray[int(c)]=1
                        print("Detecting faces!")

                    s += '%g %ss, ' % (n, names[int(c)])  # Add Tags

                bbox_xywh = []
                confs = []

                # Input the test results into the tracking
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])

                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)

                # Detection
                outputs = deepsort.update(xywhs, confss, im0)

                # Face Detection
                if startFace == True:
                    if len(outputs) != 0:
                        for j, output in enumerate(outputs):
                            if ( facesArray[j] == 1 ):
                                bbox_left = output[0]
                                bbox_top = output[1]
                                bbox_w = output[2]
                                bbox_h = output[3]
                                identity = output[-1]
                                detect_faces( face_cascade, im0, bbox_left, bbox_top, bbox_w, bbox_h, save_path)

                # Logo border
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy, identities)

                if save_txt and len(outputs) != 0:
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        identity = output[-1]

                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left, bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

            else:
                deepsort.increment_ages()

            #print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Call the camera to display the result
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save the results
            if save_img:
                if dataset.mode == 'images':
                    print('Saving AI result image!')
                    cv2.imwrite(save_path, im0)
                else:
                    print('Saving AI result video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Use albert custom dataset
    parser.add_argument('--weights', type=str,
                        default='albert.pt', help='model.pt path')
    # Data Sources
    parser.add_argument('--source', type=str,
                        default='Traffic.mp4', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='0',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    # Default vehicle categories
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[0, 1, 2, 3, 5, 6, 7, 86, 87, 88, 89, 90, 91, 92, 93], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
    # parser.add_argument("--weights_deepsort", type=str,
    #                     default="deep_sort_pytorch/deep_sort/deep/checkpoint/mars.pb")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        detect(args)
