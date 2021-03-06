# Yolov5-face for landmarks


<div align="center">

<a href="https://colab.research.google.com/drive/19dEsb7BOM8TdCKVrjRtGt9Mt0J-rvcwQ?authuser=1#scrollTo=JKeU4mjgRlbW"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>

</div>


## Introduction

The detections generated by [YOLOv5-face](https://github.com/Onlyorlov/yolov5_face).


## Before you run the tracker

1. Clone the repository recursively:

`git clone --recurse-submodules https://github.com/Onlyorlov/yolov5_face_landmarks`

If you already cloned and forgot to use `--recurse-submodules` you can run `git submodule update --init`

2. Make sure that you fulfill all the requirements: Python 3.8 or later with all [requirements.txt](https://github.com/Onlyorlov/yolov5_face_landmarks/blob/master/requirements.txt) dependencies installed, including torch>=1.7. To install, run:

`pip install -r requirements.txt`


## Tracking sources

Tracking can be run on most video formats

```bash
$ python track.py --source img.jpg  # image
                           vid.mp4  # video
                           path/  # directory
                           path/*.jpg  # glob
```

## Decrease FPS

By default frame rate will be decreased to 1 fps for increased performance.
If you want to process all frames of a video pass 0 or choose another coef to decrease fps.

```bash
python3 track.py --coef 1   # 1 fps
                        0   # without changes
                        int # step = original_fps/int
```

## Select object detection model

There is a clear trade-off between model inference speed and accuracy. In order to make it possible to fulfill your inference speed/accuracy needs
you can select a Yolov5 family model for automatic download

```bash


$ python track.py --source path/vid.mp4    --yolo_model yolov5m-face.pt
                                                        yolov5s-face.pt
                                            ...
```
[List of pretrained models with weights](https://github.com/Onlyorlov/yolov5_face)

## Select image resolution for yolo

Default: [640x640]. 
Aspect ratio remains the same(by padding with 0).?

```bash


$ python track.py --source 0 --img-size 640
                                        1280
                                            ...
```

## Saving results

If you want to save results as videos

```bash
python3 track.py --source path/vid.mp4  --yolo_model yolov5/weights/yolov5m-face.pt --save-vid
```

If you want to save results as images

```bash
python3 track.py --source path/vid.mp4  --yolo_model yolov5/weights/yolov5m-face.pt --save-photo
```
