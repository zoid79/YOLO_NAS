import cv2
import torch
from super_gradients.training import models
from super_gradients.common.object_names import Models

model = models.get('yolo_nas_s',num_classes=7,checkpoint_path='weights/yolo_nas_s')

model = model.to("cuda" if torch.cuda.is_available() else 'cpu')

model.predict_webcam()