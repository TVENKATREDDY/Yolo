import random
import cv2
import numpy as np
from ultralytics import YOLO
my_file=open(r'C:\Users\91807\VSCodeProjects\PythonPractice\11.YOLO\utils\coco.txt')
data=my_file.read()
class_list=data.split('\n')
my_file.close()
print(class_list)