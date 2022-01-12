# limit the number of cpus used by high performance libraries
import os
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
import json
import pickle
import shutil
import time
import numpy as np
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5-face.models.common import DetectMultiBackend
from yolov5-face.utils.datasets import LoadImages, LoadStreams
from yolov5-face.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                                  check_imshow, xyxy2xywh, increment_path)
from yolov5-face.utils.torch_utils import select_device, time_sync
from yolov5-face.utils.plots import Annotator, colors

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def detect(opt):
    out, source, yolo_model, show_vid, save_vid, imgsz, project, name, mask, coef, exist_ok = \
        opt.output, opt.source, opt.yolo_model, opt.show_vid, opt.save_vid, \
        opt.imgsz, opt.project, opt.name, opt.mask, opt.coef, opt.exist_ok
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = select_device(opt.device)

    # Directories
    save_dir = increment_path(Path(project) / name,
                              exist_ok=exist_ok)  # increment run
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
    stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        show_vid = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz,
                              stride=stride, mask=mask, auto=pt and not jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadMaskedImages(
            source, img_size=imgsz, stride=stride, mask=mask, coef=coef, auto=pt and not jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

    if pt and device.type != 'cpu':
        model(torch.zeros(
            1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0

    # output for total counter
    output = []
    out = []
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        t1 = time_sync()
        img = torch.from_numpy(img).to(device)
        img = img.float()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(
            save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
        pred = model(img, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if save_vid or show_vid:
                seen += 1
                if webcam:  # batch_size >= 1
                    p, im0, _ = path[i], im0s[i].copy(), dataset.count
                    s += f'{i}: '
                else:
                    p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                s += '%gx%g ' % img.shape[2:]  # print string

                annotator = Annotator(im0, line_width=2, pil=not ascii)
                p_count = 0
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(
                        img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        # add to string
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                        if int(c) == 0:
                            p_count = n.item()

                    xyxy = det[:, 0:4]
                    confs = det[:, 4]
                    clss = det[:, 5]

                    # draw boxes for visualization
                    if len(det) > 0:
                        for j, (bbox, cls, conf) in enumerate(zip(xyxy.cpu(), clss.cpu(), confs.cpu())):

                            c = int(cls)  # integer class
                            label = f'{names[c]} {conf:.2f}'
                            annotator.box_label(
                                bbox, label, color=colors(c, True))

                        if mask and p_count:
                            annotator.text(
                                [0, 0], f'{p_count} people in target region', color=colors(0, True))

                # Print time (inference-only)
                # LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s)')

                im0 = annotator.result()
                # Stream results
                if show_vid:
                    cv2.imshow(p, im0)
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

                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

            else:
                seen += 1
                if webcam:  # batch_size >= 1
                    p = path[i]
                    s += f'{i}: '
                else:
                    p = path

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...

                p_count = 0
                if det is not None and len(det):
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        # add to string
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "
                        if int(c) == 0:
                            p_count = n.item()

                # Print time (inference-only)
                # LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s)')

            out.append((path, s, p_count))  # save detections
            # Save counter
            if vid_path != save_path:  # if new video
                LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s)')
                vid_path = save_path
                # save old video data/create initial file
                with open(save_path+'.txt', 'wb') as fp:
                    pickle.dump(out, fp)
                output.extend(out)
                out = []

    # Last video
    with open(save_path+'.txt', 'wb') as fp:
        pickle.dump(out, fp)
    output.extend(out)
    out = []

    # Save total results
    with open(save_dir/'output.txt', 'wb') as fp:
        pickle.dump(output, fp)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS \
        per image at shape {(1, 3, *imgsz)}' % t)

    print('Results saved to %s' % save_dir)
    if platform == 'darwin':  # MacOS
        os.system('open ' + save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_model', nargs='+', type=str,
                        default='yolov5x.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+',
                        type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float,
                        default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true',
                        help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true',
                        help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument('--evaluate', action='store_true',
                        help='augmented inference')
    parser.add_argument('--visualize', action='store_true',
                        help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000,
                        help='maximum detection per image')
    parser.add_argument('--dnn', action='store_true',
                        help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--project', default=ROOT / 'runs',
                        help='save results to project/name')
    parser.add_argument('--name', default='exp',
                        help='save results to project/name')
    parser.add_argument('--mask', type=json.loads, default=None,
                        help='all outer corners of the region of interest(ROI)')
    parser.add_argument('--coef', type=int, default=1,
                        help='desired coef to decrease fps')
    parser.add_argument('--exist-ok', action='store_true',
                        help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    with torch.no_grad():
        detect(opt)
