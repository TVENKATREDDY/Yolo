import random
import cv2
import numpy as np
from ultralytics import YOLO
my_file=open(r'C:\Users\91807\VSCodeProjects\PythonPractice\11.YOLO\utils\coco.txt')
data=my_file.read()
class_list=data.split('\n')
my_file.close()
detection_colors=[]
for i in range(len(class_list)):
    r=random.randint(0,255)
    g=random.randint(0,255)
    b=random.randint(0,255)
    detection_colors.append((b,g,r))
    
model=YOLO("weights/yolov8n.pt","v8")
frame_wid=640
frame_hyt=480

cap=cv2.VideoCapture(r'C:\Users\91807\VSCodeProjects\PythonPractice\11.YOLO\video\video_sample1.mp4')
if not cap.isOpened():
    print('cannot open camera')
    exit()
while True:
    ret,frame=cap.read()
    if not ret:
        print("can't receive frame ...Exiting..")
        break
    detect_params=model.predict(source=[frame],conf=0.45,save=True)
    dp=detect_params[0].numpy()
    print('dp :',dp)
    if len(dp)!=0:
        for i in range(len(detect_params[0])):
            print('i is:',i)
            boxes=detect_params[0].boxes
            box=boxes[i]
            clsId=box.cls.numpy()[0]
            conf=box.conf.numpy()[0]
            bb=box.xyxy.numpy()[0]
            cv2.rectangle(
                frame,(int(bb[0]),int(bb[1])),
                (int(bb[2]),int(bb[3])),
                detection_colors[int(clsId)],
                3,
            )
            font=cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(
                frame,
                class_list[int(clsId)] + " " + str(round(conf,3)) + "%" ,
                (int(bb[0]),int(bb[1]) - 10),
                font,
                1,
                (255,255,255),
                2,
            )
    cv2.imshow('ObjectDetection',frame)
    if cv2.waitKey(1)== ord('q'):
        break
cap.release()
cv2.destroyAllWindows()