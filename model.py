
from ultralytics import YOLO
import torchvision.models.detection as detection
import torch
import supervision as sv
import numpy as np

YOLO_MODEL_TYPE = "YOLO"
FASTER_RCNN_MODEL_TYPE = "FASTER-RCNN"
RETINA_NET_MODEL_TYPE = "RETINA_NET"
class ModelProxy:
    def __init__(self, model_type:str):
        self.model_type = model_type
        if self.model_type == YOLO_MODEL_TYPE:
            self.model = YOLO("yolov8x.pt")
        else:
            if self.model_type == FASTER_RCNN_MODEL_TYPE:
                self.model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
            elif self.model_type == RETINA_NET_MODEL_TYPE:
                self.model = detection.retinanet_resnet50_fpn(pretrained=True)
            else:
                raise Exception("Model type not supported")
            self.model.eval()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)

    def infer(self, frame: np.ndarray) -> sv.Detections:
        if self.model_type == "YOLO":
            return sv.Detections.from_ultralytics(self.model(frame)[0])
        else :
            img_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()
            img_tensor /= 255.0
            img_tensor = img_tensor.unsqueeze(0)
            pred = self.model(img_tensor)[0]
            boxes = pred['boxes'].detach().cpu().numpy()
            scores = pred['scores'].detach().cpu().numpy()
            labels_temp = pred['labels'].detach().cpu().numpy()

            return sv.Detections(xyxy=boxes, confidence=scores, class_id=labels_temp)
