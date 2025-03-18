import supervision as sv
import xml.etree.ElementTree as ET
from typing import Dict
class Helper:

    def __init__(self, source, video_info, confidence_threshold = 0.5, iou_threshold = 0.7):
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source = source
        self.byte_track = sv.ByteTrack(
            frame_rate=video_info.fps, track_activation_threshold=confidence_threshold
        )
        self.validation_tracker : Dict[int, int] = {}
        thickness = sv.calculate_optimal_line_thickness(
            resolution_wh=video_info.resolution_wh
        )

        text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
        self.box_annotator = sv.BoxAnnotator(thickness=thickness)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=thickness,
            text_position=sv.Position.BOTTOM_CENTER,
        )
        self.trace_annotator = sv.TraceAnnotator(
            thickness=thickness,
            trace_length=video_info.fps * 2,
            position=sv.Position.BOTTOM_CENTER,
        )

    def filter_detections(self, detections: sv.Detections):
        polygon_zone = sv.PolygonZone(polygon=self.source)
        detections = detections[detections.confidence > self.confidence_threshold]
        detections = detections[polygon_zone.trigger(detections)]
        detections = detections.with_nms(threshold=self.iou_threshold)
        return self.byte_track.update_with_detections(detections=detections)

    def annotate_frame(self, frame,labels, detections: sv.Detections):
        annotated_frame = frame.copy()

        annotated_frame = self.trace_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = self.box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        return self.label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )

    @staticmethod
    def load_annotations__ground_truth(frame_num):
        frame_num = str(frame_num)
        nulls_to_add = 4 - len(frame_num)
        frame_num = nulls_to_add * "0" + frame_num
        xml_file = f"ground_truth/annotations/frame_{frame_num}.xml"

        tree = ET.parse(xml_file)
        root = tree.getroot()

        objects = []
        for obj in root.findall("object"):
            track_id = int(obj.find("track_id").text)
            bbox = {
                "xmin": int(obj.find("bndbox/xmin").text),
                "ymin": int(obj.find("bndbox/ymin").text),
                "xmax": int(obj.find("bndbox/xmax").text),
                "ymax": int(obj.find("bndbox/ymax").text),
            }
            objects.append({"track_id": track_id, "bbox": bbox})

        return objects


    def get_true_positive_rate(self, frame_num, detections):

        ground_truths = Helper.load_annotations__ground_truth(frame_num)

        true_poositive_num = 0

        for i in range(len(detections.xyxy)):
            x_min, y_min, x_max, y_max = detections.xyxy[i]
            det_id = detections.tracker_id[i]
            det_bbox = { "xmin" : x_min, "ymin" : y_min, "xmax": x_max, "ymax": y_max}
            best_iou = 0
            matched_gt_id = None

            for gt in ground_truths:
                gt_id, gt_bbox = gt["track_id"], gt["bbox"]
                iou = Helper.compute_iou(gt_bbox, det_bbox)

                if iou > best_iou:
                    best_iou = iou
                    matched_gt_id = gt_id

            is_tp = best_iou >= self.iou_threshold

            if det_id not in self.validation_tracker:
                self.validation_tracker[det_id] = matched_gt_id

            is_tp = is_tp and self.validation_tracker[det_id] == matched_gt_id

            true_poositive_num = true_poositive_num + 1 if is_tp else true_poositive_num


        return true_poositive_num / len(ground_truths)


    @staticmethod
    def compute_iou(gt_bbox, det_bbox):

        x_left = max(gt_bbox["xmin"], det_bbox["xmin"])
        y_top = max(gt_bbox["ymin"], det_bbox["ymin"])
        x_right = min(gt_bbox["xmax"], det_bbox["xmax"])
        y_bottom = min(gt_bbox["ymax"], det_bbox["ymax"])

        intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

        gt_area = (gt_bbox["xmax"] - gt_bbox["xmin"]) * (gt_bbox["ymax"] - gt_bbox["ymin"])
        det_area_area = (det_bbox["xmax"] - det_bbox["xmin"]) * (det_bbox["ymax"] - det_bbox["ymin"])

        union_area = gt_area + det_area_area - intersection_area

        iou = intersection_area / union_area if union_area > 0 else 0
        return iou


    def reset(self):
        self.byte_track.reset()
        self.validation_tracker = {}
