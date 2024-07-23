

# YoloV5-DeepSort

Fast object tracking with yolov5-deepsort is also equipped with a face detection feature (Gender and Age).

  

# Installation

Python 3.6 or later with all requirements.txt dependencies installed, including torch>=1.7. To install run:

```bash

pip  install  -r  requirements.txt

```

  

# Run Tracker

After you download this project, please download the weight of YOLO V5 model and Deep-SORT model respectively.

  

You can download the weight trained by me. Using email at: albertflicky@gmail.com

  

**Trainned Data**

- 0: person

- 1: bicycle

- 2: car (na)

- 3: motorcycle

- 4: airplane

- 5: bus

- 6: train

- 7: truck

- 8: boat

- 9: traffic light

- 10: fire hydrant

- 11: stop sign

- 12: parking meter

- 13: bench

- 14: bird

- 15: cat

- 16: dog

- 17: horse

- 18: sheep

- 19: cow

- 20: elephant

- 21: bear

- 22: zebra

- 23: giraffe

- 24: backpack

- 25: umbrella

- 26: handbag

- 27: tie

- 28: suitcase

- 29: frisbee

- 30: skis

- 31: snowboard

- 32: sports ball

- 33: kite

- 34: baseball bat

- 35: baseball glove

- 36: skateboard

- 37: surfboard

- 38: tennis racket

- 39: bottle

- 40: wine glass

- 41: cup

- 42: fork

- 43: knife

- 44: spoon

- 45: bowl

- 46: banana

- 47: apple

- 48: sandwich

- 49: orange

- 50: broccoli

- 51: carrot

- 52: hot dog

- 53: pizza

- 54: donut

- 55: cake

- 56: chair

- 57: couch

- 58: potted plant

- 59: bed

- 60: dining table

- 61: toilet

- 62: tv

- 63: laptop

- 64: mouse

- 65: remote

- 66: keyboard

- 67: cell phone

- 68: microwave

- 69: oven

- 70: toaster

- 71: sink

- 72: refrigerator

- 73: book

- 74: clock

- 75: vase

- 76: scissors

- 77: teddy bear

- 78: hair drier

- 79: toothbrush

- 80: fire

- 81: celurit

- 82: sword

- 84: gun

- 85: cigarette

- 86: pedicab

- 87: sport car

- 88: three wheel

- 89: ambulance

- 90: emergency vehicle

- 91: mini bus

- 92: fire trucks

- 93: license plate

- 94: container

- 95: helmet

  ![enter image description here](https://ingeninnovationtech.co.id/wp-content/uploads/2024/07/results.png)

or choose to download the pretrained weight of the YOLO V5 model with using the `./yolov5/weights/downloadweight.sh`.


Tracking can be run on most video formats

```bash

python3  track.py  --source  ...

```

  

- Video: `--source file.mp4`

- Webcam: `--source 0`

- RTSP stream: `--source rtsp://192.168.1.2/live`

  

MOT compliant results can be saved to `inference/output` by

  

```bash

python3  track.py  --source  ...  --save-txt

```

**Face Detection Result**

All detected face is saved to `inference/output` with filename format : `xxxx_[face_number]_[gender].jpg`
  

# Train YOLO weights with your datasets

Please have a reference on https://github.com/ultralytics/yolov5.

  

# References

https://github.com/ultralytics/yolov5
