from ultralytics import YOLO
import numpy
model=YOLO('yolov8n.pt','v8')
detect=model.predict(source=r'C:\Users\91807\VSCodeProjects\PythonPractice\11.YOLO\img\1.jpg')
print('tensor array: ',detect)
print('numpy array: ',detect[0].numpy())